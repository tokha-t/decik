# Soup Kernel Challenge — решение

Тот же `soup train` (Qwen2.5‑1.5B NF4, LoRA r=16 на q/v, 64 шага, Kaggle T4), но быстрее и с меньшим
пиком памяти. Всё подключается одной функцией `solution.apply()`; задача, данные, батчи, LoRA,
оптимизатор и выходы Soup не меняются.

- **Kaggle-ноутбук (публичный, сохранённая версия):** `<ссылка>`
- **Финальный коммит:** `<ссылка на коммит>`

## Результаты (Kaggle T4, `harness/run_task.py`, публичные строки)

| | База Soup 0.77.0 | Наше решение | Отношение |
|---|---|---|---|
| Время `soup train`, с | `<base>` | `<ours>` | speedup `<s>`× |
| Пик памяти GPU (nvidia-smi), МиБ | `<base>` | `<ours>` | base/ours `<m>`× |
| Eval loss (строки 40000–40255) | `<base>` | `<ours>` | `<Δ>` % |
| Доля наших кернелов (проверка 4) | — | `<x>` % | ≥ 5 % |

Все 10 проверок `harness/check_rules.py`: `<PASS / какие нет>`.

## Куда уходили память и время в базе (разбор кода Soup / transformers / bitsandbytes)

1. `prepare_model_for_kbit_training` поднимает эмбеддинг (151 936 × 1536) в fp32, поэтому весь
   residual-стрим fp32. `bitsandbytes.Linear4bit` считает матмул в fp16, но возвращает результат в
   dtype входа, то есть в fp32. В итоге gate, silu(gate) и up в MLP хранятся для backward в fp32:
   105 КБ на токен на слой, около 5,9 ГБ на батче 4×512. При этом их значения точно представимы в fp16.
2. Логиты [токены × 151 936] в fp16, их fp32-копия для loss, `log_softmax` для backward, плотный
   fp32-градиент и его fp16-копия занимают около 3,5 ГБ на батче 4×512. Плюс TRL ещё раз читает логиты
   для энтропии и accuracy, а Soup — для rewind.
3. Эмбеддинг в fp32 занимает 890 МиБ, и autocast каждый forward делает его fp16-копию (445 МиБ).
   Она живёт до backward.
4. Паддинг. TRL дополняет батч справа до самой длинной строки; на этих данных 44 % токен-слотов
   шага — паддинг (в среднем 826 реальных токенов против 1 467 слотов).
5. 392 распаковки NF4 за шаг (7 линейных слоёв × 28 слоёв × forward и backward) и fp32-нормы
   из шести операций каждая.

## Что сделано

| Модуль | Что делает | Кернел | Точность |
|---|---|---|---|
| `solution/nf4.py` | Распаковка NF4 → fp16 внутри каждого 4-битного линейного слоя, forward и backward (как у bitsandbytes: деквант + `F.linear`) | `skc_nf4_dequant_kernel` | бит-в-бит с bitsandbytes |
| `solution/rmsnorm.py` | `Qwen2RMSNorm` — один запуск вперёд и один назад вместо 6 + 6 операций; для backward хранится только x и один rstd на токен | `skc_rmsnorm_fwd/bwd_kernel` | до округления fp32 |
| `solution/mlp.py` | MLP с fp16-промежуточными: те же значения, втрое меньше памяти, нет fp32-кастов | (SwiGLU — кернел из примера, он не засчитывается) | бит-в-бит |
| `solution/embedding.py` | Эмбеддинг / lm_head остаются fp16 (значения и так fp16); результат поиска сразу переводится в fp32 | — | бит-в-бит |
| `solution/head.py` | lm_head + loss + метрики TRL + лоссы для rewind считаются чанками по 512 токенов, без тензора логитов | `skc_head_stats_kernel` | до округления |
| `solution/varlen.py` | Декодер без паддинга: реальные токены одной последовательностью, RoPE по позициям строки, атеншн по строкам | — | те же случайные маски dropout |
| `solution/attn.py` | LoRA-add, RoPE и перевод q/k в fp16 за один запуск; транспонирование в формат attention | `skc_rope_fwd/bwd_kernel` | выходы и градиенты бит-в-бит с неслитым PyTorch-путём в GPU-тестах |
| `solution/warmup.py` | Компиляция Triton-кернелов в фоновом потоке, пока Soup грузит модель | — | — |

### Как устроены кернелы

**`skc_nf4_dequant_kernel`.** Одна программа распаковывает 1024 байта, то есть 2048 весов.
- Старший полубайт — первый вес. Оба веса байта лежат в одном блоке absmax (64 веса).
- absmax при double quant восстанавливается как `code2[q] * absmax2[блок/256] + offset`. Умножение и
  сложение выполняются раздельно (`enable_fp_fusion=False`), как у bitsandbytes.
- Вес = `NF4[полубайт] * absmax` в fp32, затем округление в fp16.
- Пары записываются через `tl.interleave`.

**`skc_rmsnorm_fwd/bwd_kernel`.** Одна строка на программу, всё в fp32.
- Forward: `y = w * (x * rsqrt(mean(x²) + eps))`.
- Backward: `dx = rstd * (g - x * rstd² * mean(g * x))`, где `g = dy * w`. Вес нормы заморожен,
  его градиент не нужен.

**`skc_head_stats_kernel`.** Одна строка логитов чанка (fp16, их считает cuBLAS) на программу.
1. Первый проход по словарю блоками по 2048 «дорожек». На каждой дорожке ведутся онлайн-максимум,
   сумма exp и сумма z·exp, а также индекс первого максимума. Из этого получаются logsumexp,
   энтропия `lse − Σp·z` (TRL считает её по всем реальным токенам), argmax с первым индексом, как у
   `torch.argmax` (для accuracy), и лосс таргета.
2. Второй проход — только для строк с лейблом: прямо в буфер логитов записывается `softmax − onehot`
   в fp16.
3. Ещё один матмул `g @ W` даёт градиент скрытых состояний. Он хранится в fp32 и в backward умножается
   на приходящий градиент: это fp16 loss scale и `1/num_items_in_batch`.

Токены с лейблом сортируются в начало, поэтому этот матмул идёт только по ним (62 % реальных токенов).
Единственный тензор размером со словарь — буфер чанка 512 × 151 936 fp16 (148 МиБ).

**Метрики** логируются ровно по формулам TRL (`entropy`, `mean_token_accuracy`, `num_tokens`).
Лоссы токенов уходят в rewind-рекордер Soup через его же хук для шагов без логитов:
`_soup_chunked_step.token_loss` читает `soup_cli/trainer/rewind_hf.py`.

**Без паддинга.** Позиции строки 0..len−1 дают тот же RoPE, causal-атеншн по токенам строки — ту же
математику, что батч с маской даёт для реальных позиций. Маска dropout LoRA тянется той же операцией
на той же форме [B, S, H], что и в базе, и из неё берутся строки реальных токенов. Поэтому обучение
идёт по тем же случайным числам, что у базы.

### Источники идей (код наш)
- Liger Kernel, fused linear cross entropy (градиент считается в forward, чанками):
  https://github.com/linkedin/Liger-Kernel
- Cut Your Losses in Large-Vocabulary Language Models (Apple, CCE — не материализовать логиты):
  https://arxiv.org/abs/2411.09009
- QLoRA и формат NF4 / double quant: https://arxiv.org/abs/2305.14314 и исходники bitsandbytes
- Padding-free обучение (позиции по строкам, атеншн по строкам):
  https://huggingface.co/blog/packing-with-FA2
- Triton tutorials (layer norm, fused softmax): https://triton-lang.org/main/getting-started/tutorials/

## Как запустить

1. Kaggle: New Notebook → File → Import Notebook → `templates/kaggle_notebook.ipynb`. Accelerator
   **GPU T4 x2**, Internet **on**.
2. В первой ячейке: `REPO_URL` = этот репозиторий, `COMMIT` = финальный коммит.
3. **Run All**. Ноутбук ставит пакеты, гоняет `pytest solution` (наши тесты против bitsandbytes и
   PyTorch), затем `harness/run_task.py`, `harness/check_rules.py` и `harness/score.py`.

Локально на Linux с CUDA-GPU — как в стартовом ките:

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q
python harness/run_task.py --repeats 2
python harness/check_rules.py
```

Для A/B-сравнений любую часть можно выключить переменной окружения: `SKC_NF4`, `SKC_RMSNORM`,
`SKC_MLP`, `SKC_EMBED`, `SKC_HEAD`, `SKC_VARLEN`, `SKC_ATTN`, `SKC_WARMUP` = `0`.
По умолчанию включён пересчёт gate/up MLP в backward: меньший пик памяти ценой двух дополнительных
проекций на слой. `SKC_RECOMPUTE=0` выбирает более быстрый вариант с сохранением активаций.
Данные, порядок RNG и настройки тренера сохраняются. Какой вариант выгоднее по
баллам, зависит от лучших speedup и memory ratio среди допущенных команд; меньше памяти само по себе
не гарантирует больше баллов.

В `solution/` есть 23 GPU-теста: NF4, RMSNorm, SwiGLU, голова, слитый RoPE и MLP с пересчётом.
`kaggle_run.ipynb` запускает их, затем по два прогона базы и решения, после чего все 10 проверок.
Любая ошибка останавливает ноутбук. `validation/result.json`, `validation/checks.json` и
`validation/manifest.json` сохраняют результаты и точный коммит.

## Ограничения и что дальше

- Атеншн по строкам — цикл из 4 вызовов SDPA на слой. Варлен-кернел на Triton убрал бы цикл, но
  атеншн занимает около 5 % шага.
- Распаковка NF4 пишет полную fp16-копию веса перед cuBLAS. Слитый NF4-матмул в Triton сэкономил бы
  этот проход, но на T4 cuBLAS быстрее Triton-матмула.
- Пик памяти определяет самый длинный батч. Следующий шаг по памяти — пересчёт gate/up в backward или
  асинхронный офлоад активаций в CPU: −1,5…2,5 ГБ ценой 0–6 % скорости.
- `<что не получилось / где решение слабее — дописать по итогам прогонов>`
