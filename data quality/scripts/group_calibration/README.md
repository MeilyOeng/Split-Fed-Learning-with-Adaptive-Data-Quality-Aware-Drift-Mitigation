# Group Calibration

This folder contains a standalone script for calibrating aggregated demand behavior by metadata groups.

It does four things:

- matches wide demand columns to metadata `ID`
- aggregates all clients into `Categoria` and `Subcategoria`
- builds both total aggregated demand and per-client average demand series
- writes histogram, tail, percentile, and min-max range outputs directly to `outputs/calibrations/`

Default command:

```bash
python "data quality/scripts/group_calibration/calibrate_groups.py" \
  --demand dataset/dataset.csv \
  --metadata dataset/general_categories.csv \
  --timestamp-col Fecha_Hora
```

Main outputs:

- `outputs/calibrations/matching_report.json`
- `outputs/calibrations/category/aggregated/category_aggregated_timeseries.csv`
- `outputs/calibrations/category/aggregated/category_aggregated_summary.csv`
- `outputs/calibrations/category/aggregated/category_aggregated_range_overview.png`
- `outputs/calibrations/category/aggregated/groups/*.png`
- `outputs/calibrations/category/per_client_average/category_per_client_average_timeseries.csv`
- `outputs/calibrations/category/per_client_average/category_per_client_average_summary.csv`
- `outputs/calibrations/category/per_client_average/category_per_client_average_range_overview.png`
- `outputs/calibrations/category/per_client_average/groups/*.png`
- `outputs/calibrations/subcategory/aggregated/subcategory_aggregated_timeseries.csv`
- `outputs/calibrations/subcategory/aggregated/subcategory_aggregated_summary.csv`
- `outputs/calibrations/subcategory/aggregated/subcategory_aggregated_range_overview.png`
- `outputs/calibrations/subcategory/aggregated/groups/*.png`
- `outputs/calibrations/subcategory/per_client_average/subcategory_per_client_average_timeseries.csv`
- `outputs/calibrations/subcategory/per_client_average/subcategory_per_client_average_summary.csv`
- `outputs/calibrations/subcategory/per_client_average/subcategory_per_client_average_range_overview.png`
- `outputs/calibrations/subcategory/per_client_average/groups/*.png`
