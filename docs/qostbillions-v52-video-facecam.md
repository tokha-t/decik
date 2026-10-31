# Демо-видео v5.2: лицо в углу поверх экрана (до 3:00)

Формат: экранное видео (слайды + публичный прогон Kaggle + код на GitHub) на весь кадр, лицо Токтархана в правом
нижнем углу, как у стримеров. Экранную часть собирает GPT по сценарию ниже; голос и лицо записывает Токтархан.
Говорим «на экране», а не «сейчас я нажимаю»: экран показывает сохранённые результаты, а не живую работу.

## 1. Текст для телесуфлёра (читать вслух, ≈2:55)

**0:00–0:20 · слайд 1**
Привет! Мы — команда QostBILLions. Мы ускорили дообучение Qwen2.5-1.5B на Kaggle T4: те же данные, та же LoRA,
те же 64 шага. Финальная версия пять-два: обучение быстрее в 1,43 раза, пик памяти меньше в 7,08 раза,
все десять проверок пройдены.

**0:20–0:45 · слайд 2**
В базовом прогоне память съедают промежуточные тензоры MLP в fp32, полные логиты по словарю из 152 тысяч
токенов и копии эмбеддинга. А 44 процента токен-слотов — это паддинг. Эти расходы мы и убираем, не меняя
саму задачу обучения.

**0:45–1:15 · слайд 3**
Убираем паддинг внутри батча и сохраняем те же случайные маски dropout. Атеншн всех строк считаем одним
вызовом на слой; для обратного прохода храним только Q, K и V и повторяем forward. Входы LoRA и MLP
пересчитываем, маски храним битами. Вход нормы после атеншна восстанавливаем из residual и выхода атеншна. Голову, loss
и метрики считаем чанками, без полной матрицы логитов.

**1:15–1:45 · слайд 4, затем код на GitHub**
В обучении работают девять наших Triton-кернелов: распаковка NF4, RMSNorm в обе стороны, RoPE, маски и
входы LoRA, статистики головы. На экране — их код в папке solution. Матмулы выполняет cuBLAS, а пример
SwiGLU из стартового набора честно указан и не засчитывается.

**1:45–2:15 · слайд 5, затем Kaggle**
База и решение измерены в одной сессии, по два прогона. Время — с 87,9 до 61,3 секунды, память — с 14 345
до 2 025 мегабайт, компиляция включена во время. Сам цикл обучения быстрее в 1,61 раза. На экране —
публичный сохранённый прогон на Kaggle: eval loss совпал с базой.

**2:15–2:40 · слайд 6, затем список проверок**
Корректность проверяют 22 GPU-теста и все 79 регрессионных тестов Soup. Все десять правил допуска
пройдены: те же шаги, токены, метрики и адаптер. Наши кернелы занимают 13,4 процента времени обучения
при пороге в пять.

**2:40–2:55 · слайд 7 и финальная карточка**
Пересчёт экономит память ценой вычислений — это осознанный компромисс. Код, ноутбук и все результаты —
по ссылкам на экране. Спасибо!

## 2. Сценарий экранного видео для GPT

Output: `screen_v52.mp4`, 1920×1080, 30 fps, H.264, no audio, **2:57** (177 s). The bottom-right corner
(440×330 px + 32 px margin) will be covered by the facecam: keep important numbers out of it.

| Time | Length | What is on screen | Source |
|---|---:|---|---|
| 0:00–0:20 | 20 s | Slide 1 | `docs/qostbillions-v52-final-deck.pdf`, page 1 |
| 0:20–0:45 | 25 s | Slide 2 | page 2 |
| 0:45–1:15 | 30 s | Slide 3 | page 3 |
| 1:15–1:33 | 18 s | Slide 4 | page 4 |
| 1:33–1:45 | 12 s | GitHub: the `solution/` folder (file list), then `solution/nf4.py` scrolled to `@triton.jit` / `skc_nf4_dequant_kernel` | `https://github.com/tokha-t/decik/tree/8414ab48cdc23c2ddb509f5418f41c2457d40811/solution` |
| 1:45–2:03 | 18 s | Slide 5 | page 5 |
| 2:03–2:15 | 12 s | Kaggle log: the lines `base … solution … speedup 1.433x memory base/solution 7.084x eval loss vs base 0.0 %` and `ALL CHECKS PASSED`; slow zoom on them | `https://www.kaggle.com/code/toktarkhan/notebook72649ae047/log?scriptVersionId=364288750` |
| 2:15–2:30 | 15 s | Slide 6 | page 6 |
| 2:30–2:40 | 10 s | Kaggle log: the ten `PASS 1.` … `PASS 10.` lines (check 4 shows 13.4%) | same log page |
| 2:40–2:51 | 11 s | Slide 7 | page 7 |
| 2:51–2:57 | 6 s | End card on the deck's dark background: "QostBILLions · Soup Kernel Challenge", repository link, Kaggle notebook link, "v5.2 · 8414ab4" | make it |

How to build it:
- Slides: `pdftoppm -png -scale-to-x 1920 -scale-to-y 1080 docs/qostbillions-v52-final-deck.pdf slide`.
- Web pages: Playwright screenshots at 1920×1080, browser zoom 125–150 % so text reads on a phone; logged-out
  view (the notebook is public); hide cookie banners; light, clean pages, no bookmarks bar.
- Assemble with ffmpeg: one image per row with its exact length, a 0.3 s cross-fade between rows, a slow
  `zoompan` (≤ 1.1×) only on the Kaggle and GitHub shots. Check the total is 177 s.
- Also export `screen_v52_preview.mp4` with the row timestamps burned in small at the top-left (for rehearsal
  only, not for the final cut).

## 3. Запись и сборка

**Вариант A — одним дублем в OBS (рекомендую, синхронизация не нужна).** Холст 1920×1080. Источник
«Медиа» = `screen_v52.mp4` на весь кадр (включить «Перезапускать при активации»). Источник «Камера»
в правом нижнем углу, 440×330, отступ 32 px. Микрофон. Нажать «Начать запись», переключиться на сцену —
видео стартует с начала, читать текст по таймингу. Перезаписать, если сбился: 3–4 дубля нормально.

**Вариант B — отдельно лицо, потом склейка.** Записать лицо (горизонтально, 1080p, свет спереди, камера на
уровне глаз), читая текст, пока `screen_v52_preview.mp4` идёт перед глазами. Склеить в CapCut (картинка в
картинке) или командой (GPT подставит сдвиг `OFFSET`, чтобы первое слово совпало с 0:00):

```bash
ffmpeg -i screen_v52.mp4 -ss OFFSET -i face.mp4 -filter_complex \
  "[1:v]crop=ih*4/3:ih,scale=440:330,format=yuv420p[cam];[0:v][cam]overlay=W-w-32:H-h-32[v]" \
  -map "[v]" -map 1:a -c:v libx264 -preset veryfast -crf 20 -c:a aac -b:a 160k -t 179 qostbillions_demo_v52.mp4
```

**Проверить перед загрузкой:** длительность ≤ 3:00; лицо не закрывает цифры; звук без эха; числа в речи
совпадают со слайдами (1,43× · 7,08× · 2 025 МиБ · 22 и 79 тестов · 13,4 %). Загрузить на YouTube
(«доступ по ссылке») или Google Drive («все, у кого есть ссылка»), открыть ссылку в режиме инкогнито и
вставить её в `SUBMISSION.md` (Demo video) и в README.
