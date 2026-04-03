# Column mapping

The raw Excel files in this repository keep the original Chinese column names used during data collection. The code in this repository reads these original field names directly.

For convenience, the corresponding English meanings are listed below.

| Chinese column name | English meaning | Suggested English variable name |
|---|---|---|
| 日期 | Date | date |
| 耳缺号 | Ear tag ID / Pig ID | ear_tag_id |
| 采食量 | Feed intake | feed_intake |
| 采食次数 | Feeding frequency | feeding_frequency |
| 采食时间 | Feeding duration | feeding_duration |
| 体重 | Body weight | body_weight |
| 日龄 | Age in days | age |
| env_NH3 | Environmental ammonia concentration | env_nh3 |
| v_二氧化碳 | Environmental carbon dioxide concentration | env_co2 |
| env_温度 | Environmental temperature | env_temperature |
| env_湿度 | Environmental relative humidity | env_humidity |

## Notes

1. The repository keeps the original Chinese field names to remain consistent with the raw experimental records and the provided code.
2. The English names above are provided only as references for international readers.
3. In the current code, the raw Chinese column names are used directly when reading and processing the Excel files.
