# دستورالعمل داور مستقل — SWE-QA v4

## هدف

شما داور مستقل کیفیت پاسخ‌های سیستم پرسش‌وپاسخ کد هستید. ارزیابی را کاملاً مستقل انجام دهید و از نتایج داور دیگر، automatic judge یا مدل زبانی برای تولید امتیاز استفاده نکنید.

نمونه شامل ۳۰ سؤال از چهار repository است: Flask (۸)، Requests (۸)، pytest (۷) و Pylint (۷). برای هر سؤال سه پاسخ ناشناس در Task 1 وجود دارد؛ در مجموع ۹۰ ردیف.

## فایل‌های مورد استفاده

فقط این نسخه‌های مستقل را ویرایش کنید:

- `reports/evaluation/swe_qa_v4/human_review_v4/task1_items_independent.csv`
- `reports/evaluation/swe_qa_v4/human_review_v4/task2_items_independent.csv`

برای مشاهده‌ی کد citationها در Task 2 از این فایل استفاده کنید:

- `reports/evaluation/swe_qa_v4/human_review_v4/task2_cited_code.csv`

راهنمای معیارها:

- `reports/evaluation/swe_qa_v4/HUMAN_REVIEW_RUBRIC.md`
- `reports/evaluation/swe_qa_v4/HUMAN_REVIEW_PACKAGE_SCHEMA.md`

فایل‌های `task1_items.csv` و `task2_items.csv` نسخه‌ی امتیازدهی‌شده‌ی داور دیگر هستند؛ آن‌ها را ویرایش نکنید.

## دسترسی به کد واقعی

برای امتیازدهی groundedness، در صورت نیاز snapshot همان repository را بررسی کنید:

```text
data/benchmarks/swe_qa_v4/repos/flask
data/benchmarks/swe_qa_v4/repos/requests
data/benchmarks/swe_qa_v4/repos/pytest
data/benchmarks/swe_qa_v4/repos/pylint
```

commitهای frozen:

```text
flask    85c5d93cbd049c4bd0679c36fd1ddcae8c37b642
requests 46e939b5525d9c72b677340985582b04b128478a
pytest   5989efe3efec31a9d14f550d89903b6311dfc828
pylint   44740e5e5cdd2b80321a4f0ed4254213d058b9f6
```

اگر پاسخ درباره‌ی فایل، تابع، کلاس، symbol یا line range ادعایی دارد، همان فایل و محدوده را بررسی کنید؛ مطالعه‌ی کل repository لازم نیست.

## Task 1 — کیفیت پاسخ

فایل: `task1_items_independent.csv`

برای هر ۹۰ ردیف این ستون‌ها را پر کنید:

- `correctness`: درستی ادعاها، عدد صحیح ۰ تا ۱۰
- `completeness`: پوشش نکات اصلی reference answer، عدد صحیح ۰ تا ۱۰
- `relevance`: ارتباط مستقیم با سؤال، عدد صحیح ۰ تا ۱۰
- `groundedness`: قابل‌تأییدبودن ادعاها در repository، عدد صحیح ۰ تا ۱۰
- `refusal`: مقدار `yes` یا `no`
- `comment`: اختیاری

امتیاز کلی را وارد نکنید؛ میانگین چهار معیار خودکار محاسبه می‌شود. طول پاسخ و لحن مطمئن معیار امتیازدهی نیست. پاسخ کوتاه ولی دقیق می‌تواند نمره‌ی بالا بگیرد. اگر پاسخ اعلام کمبود شواهد می‌کند، آن را در چهار معیار ارزیابی کنید و در صورت امتناع واقعی `refusal=yes` بزنید.

## Task 2 — پشتیبانی citation

فایل: `task2_items_independent.csv`

هر ردیف یک citation است؛ مجموعاً ۱۳۳ citation وجود دارد. برای متن واقعی citation، مقدار `citation_id` را در `task2_cited_code.csv` پیدا کنید.

در ستون `support` یکی از این مقادیر را وارد کنید:

- `yes`: محدوده‌ی کد ادعای پاسخ را پشتیبانی می‌کند.
- `partial`: فقط بخشی از ادعا یا نکته‌ای فرعی را پشتیبانی می‌کند.
- `no`: به ادعا مربوط نیست یا آن را پشتیبانی نمی‌کند.
- `not_applicable`: پاسخ ادعای قابل استناد ندارد.

در ستون `missing_key_source` برای هر سؤال و هر label یکی از این مقادیر را وارد کنید:

- `not_applicable`: ادعای مشخصی برای بررسی منبع وجود ندارد.
- `source_identified`: citationها sourceهای لازم برای ادعاهای اصلی را دارند.
- `source_missing`: حداقل یک source مهم برای ادعای اصلی در citationها وجود ندارد.

در Task 2 دو citation list با برچسب‌های `X` و `Y` وجود دارد. نباید بدانید X مربوط به A1 است یا A2؛ نگاشت را حدس نزنید و تغییر ندهید.

## قواعد استقلال و blind

- از automatic judge یا امتیازهای قبلی استفاده نکنید.
- قبل از پایان کار با داور دیگر درباره‌ی امتیازها صحبت نکنید.
- از LLM یا ابزار تولید امتیاز استفاده نکنید.
- `item_id`، `question_id`، `system_label`، سؤال، پاسخ‌ها و citationها را تغییر ندهید.
- ردیف‌ها را حذف، اضافه یا ادغام نکنید.
- فایل‌های امتیازدهی‌شده‌ی داور دیگر را تغییر ندهید.

## خروجی نهایی

پس از پایان، دو فایل تکمیل‌شده را با این نام‌ها ذخیره کنید:

```text
task1_items_independent_scored.csv
task2_items_independent_scored.csv
```

ساختار CSV و encoding UTF-8 را حفظ کنید و فقط همین دو فایل را تحویل دهید. پس از دریافت آن‌ها، validation و تحلیل آماری شامل weighted kappa، Spearman، paired bootstrap، citation precision و claim coverage انجام می‌شود.
