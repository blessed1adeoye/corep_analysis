"""
One-Shot Full Pipeline
Runs every module in the correct order.
    python run_all.py
"""

import time
import main_analysis as base
import clinical_questions as q
import report_generator as report
import predictive_models as ml
import forecasting
import clustering
import nlp_analysis


def main():
    t0 = time.time()
    print("\n" + "=" * 60)
    print(" COREP FULL ANALYTICS PIPELINE")
    print("=" * 60)

    # 1. Load + clean once, share with all modules
    print("\n[1/7] Loading data ...")
    data = base.clean_all(base.load_data())

    # 2. Base EDA + charts
    print("\n[2/7] Running base EDA ...")
    base.chart_patient_demographics(data["patient"])
    base.chart_registration_trends(data["patient"])
    base.chart_vitals(data["nursing"])
    # NOTE: pass `data` so diagnosis normalization is used
    base.chart_consultations(data["consultations"], data["patient"], data)
    base.chart_lab_tests(data["lab_tests"])
    base.chart_optical(data["optical"])
    base.chart_pharmacy(data["pharmacy"], data["drugs"])
    base.chart_patient_journey(data["patient"], data["consultations"],
                                data["lab_tests"], data["pharmacy"], data["optical"])

    # 3. Clinical Q&A
    print("\n[3/7] Clinical questions ...")
    q.run_all(data["consultations"], data["patient"],
              data["lab_tests"], data["pharmacy"])

    # 4. ML models
    print("\n[4/7] Training ML models ...")
    ml.model_admission(data["patient"], data["consultations"], data["nursing"])
    ml.model_malaria(data["lab_tests"], data["patient"], data["nursing"])

    # 5. Forecasting
    print("\n[5/7] Forecasting patient volume ...")
    forecasting.run(data["patient"], data["consultations"])

    # 6. Clustering — FIXED: pass all 5 arguments
    print("\n[6/7] Clustering patients ...")
    clustering.run(
        data["patient"],
        data["consultations"],
        data["nursing"],
        data["lab_tests"],
        data["pharmacy"],
    )

    # 7. NLP
    print("\n[7/7] NLP on clinical notes ...")
    nlp_analysis.run(data["consultations"], data["nursing"], data["lab_tests"])

    # Final report
    print("\n📄 Building HTML report ...")
    report.generate_report()

    print(f"\n✅ PIPELINE DONE in {time.time() - t0:.1f}s")
    print("Outputs:")
    print("  • ./charts/   (all PNG charts)")
    print("  • ./reports/  (HTML/PDF report + CSVs)")
    print("  • ./models/   (trained ML models)")
    print("\nNext steps:")
    print("  • streamlit run dashboard_app.py")
    print("  • uvicorn api.app:app --reload --port 8000")


if __name__ == "__main__":
    main()