"""Run the analysis from the processed tables to the final figures, in order. Upstream download scripts (00b to 06) are not re-run."""
import subprocess, sys
from pathlib import Path

here = Path(__file__).parent
for s in ["00_setup", "04_region_assignment", "07_cohort_characteristics", "08_model_evaluation", "09_bootstrap_ci", "10_feature_importance",
          "11_baseline_models", "12_region_sensitivity", "21_post_hoc_analyses", "13_model_diagnostics", "14_study_flow_figure", "15_validation_figure",
          "16_loo_region_figure", "17_feature_importance_figure", "18_pca_mlst_figure", "20_pca_region_figure", "19_consolidate_results"]:
    print(">>", s, flush=True); subprocess.run([sys.executable, str(here / f"{s}.py")], check=True)
