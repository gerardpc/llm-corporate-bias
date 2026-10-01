"""This module is for saving the figures of the temporal discounting experiment."""

from bias_in_llms.config.project_paths import DATA_DIR
from bias_in_llms.database import select_all_from_table
from bias_in_llms.utils.data_visualization import (
    plot_order_bias,
    plot_preference_frequency,
)
from bias_in_llms.utils.utils import save_plots_path

if __name__ == "__main__":
    db_path = DATA_DIR / "temporal_discounting_results.db"
    print(f"📊 Loading data from {db_path}")
    data = select_all_from_table(
        table_name="temporal_discounting_results",
        db_path=db_path,
    )

    if data.empty:
        print("❌ No data found in database. Run the experiment first!")
        exit(1)

    print(f"✅ Loaded {len(data)} rows of data")
    models = data["model_id"].unique()
    print(f"📈 Generating plots for {len(models)} models")
    print(f"   • Models: {', '.join(models)}")
    print()

    for model_id in models:
        print(f"🎨 Processing model: {model_id}")
        save_path = save_plots_path(
            experiment_name="temporal_discounting",
            model_id=model_id,
        )
        print(f"   • Save path: {save_path}")

        model_data = data[data["model_id"] == model_id].copy()
        successful_data = model_data[model_data["answer"] != "ERROR"]

        if successful_data.empty:
            print(f"   ⚠️  Skipping model {model_id}: No successful results found.")
            print()
            continue

        print(f"   • Data points: {len(successful_data)} successful results")
        plot_preference_frequency(successful_data, model_id, save_path)
        plot_order_bias(successful_data, model_id, save_path)
        print(f"   ✅ Generated 2 plots for {model_id}")
        print()

    print("🎉 All visualizations completed!")
