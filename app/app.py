
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

from pathlib import Path
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Student Academic Risk Prediction",
    page_icon="🎓",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "notebooks" / "student_performance_model.pkl"
PREPROCESSOR_PATH = PROJECT_DIR / "notebooks" / "student_preprocessor.pkl"
DATA_PATH = PROJECT_DIR / "data" / "raw" / "StudentPerformanceFactors.csv"

NUMERICAL_FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity"
]

CATEGORICAL_FEATURES = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Motivation_Level",
    "Internet_Access",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender"
]


@st.cache_resource
def load_project():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Exam_Score"])
    y = df["Exam_Score"]

    X_train, _ = train_test_split(
        X,
        test_size=0.2,
        random_state=42
    )

    feature_names = X.columns.tolist()

    category_maps = {}
    category_names = {}

    X_train_lime = X_train.copy()

    for col in CATEGORICAL_FEATURES:
        categories = sorted(X[col].dropna().unique().tolist())

        category_maps[col] = {
            value: index
            for index, value in enumerate(categories)
        }

        category_names[col] = {
            index: value
            for index, value in enumerate(categories)
        }

        mode_value = X_train[col].mode().iloc[0]

        X_train_lime[col] = (
            X_train_lime[col]
            .fillna(mode_value)
            .map(category_maps[col])
        )

    for col in NUMERICAL_FEATURES:
        X_train_lime[col] = X_train_lime[col].fillna(
            X_train[col].median()
        )

    X_train_lime = X_train_lime[feature_names].astype(float)

    categorical_indices = [
        feature_names.index(col)
        for col in CATEGORICAL_FEATURES
    ]

    categorical_names = {
        feature_names.index(col): [
            category_names[col][i]
            for i in sorted(category_names[col])
        ]
        for col in CATEGORICAL_FEATURES
    }

    lime_explainer = LimeTabularExplainer(
        training_data=X_train_lime.to_numpy(),
        feature_names=feature_names,
        categorical_features=categorical_indices,
        categorical_names=categorical_names,
        mode="regression",
        discretize_continuous=True,
        random_state=42
    )

    shap_explainer = shap.TreeExplainer(model)

    return (
        model,
        preprocessor,
        df,
        X,
        X_train,
        feature_names,
        category_maps,
        category_names,
        lime_explainer,
        shap_explainer
    )


def classify_risk(score):
    if score < 60:
        return "High Risk"
    if score < 65:
        return "Moderate Risk"
    return "Lower Predicted Risk"


def build_suggestions(student, feature_contributions):
    suggestions = []
    ordered_features = sorted(
        feature_contributions.items(),
        key=lambda item: item[1]
    )

    suggestion_rules = {
        "Attendance": (
            lambda s: s["Attendance"] < 75,
            "Improve attendance",
            "Attend classes regularly and work toward better attendance."
        ),
        "Hours_Studied": (
            lambda s: s["Hours_Studied"] < 15,
            "Build a study routine",
            "Set aside regular study time and break assignments into manageable tasks."
        ),
        "Previous_Scores": (
            lambda s: s["Previous_Scores"] < 65,
            "Strengthen foundational topics",
            "Review difficult concepts and practise questions from previous examinations."
        ),
        "Tutoring_Sessions": (
            lambda s: s["Tutoring_Sessions"] < 2,
            "Seek academic support",
            "Ask a teacher for help with difficult topics or consider extra doubt-clearing sessions."
        ),
        "Sleep_Hours": (
            lambda s: s["Sleep_Hours"] < 7,
            "Maintain a healthy sleep routine",
            "Aim for a regular, sufficient sleep schedule to support learning."
        ),
        "Access_to_Resources": (
            lambda s: s["Access_to_Resources"] == "Low",
            "Use available learning resources",
            "Explore textbooks, library materials, practice questions and suitable learning resources."
        ),
        "Motivation_Level": (
            lambda s: s["Motivation_Level"] == "Low",
            "Set achievable learning goals",
            "Set small weekly goals and track your progress to build consistent study habits."
        ),
        "Parental_Involvement": (
            lambda s: s["Parental_Involvement"] == "Low",
            "Consider supportive check-ins",
            "Where helpful, discuss learning goals and progress with a trusted adult or teacher."
        ),
        "Teacher_Quality": (
            lambda s: s["Teacher_Quality"] == "Low",
            "Clarify difficult topics",
            "Ask for clarification, additional examples or support when lessons are difficult to follow."
        ),
        "Internet_Access": (
            lambda s: s["Internet_Access"] == "No",
            "Explore offline study options",
            "Use available textbooks, printed notes and offline practice materials."
        )
    }

    for feature, contribution in ordered_features:
        if contribution >= 0:
            continue

        rule = suggestion_rules.get(feature)

        if rule is None:
            continue

        condition, title, message = rule

        if condition(student):
            suggestions.append((title, message))

        if len(suggestions) == 5:
            break

    if not suggestions:
        suggestions.append((
            "Maintain consistent learning habits",
            "Continue revising, completing assignments and checking your understanding regularly."
        ))

    return suggestions


def explain_lime_rule(rule):
    readable_names = {
        "Hours_Studied": "Weekly study hours",
        "Attendance": "Attendance",
        "Sleep_Hours": "Daily sleep duration",
        "Previous_Scores": "Previous examination scores",
        "Tutoring_Sessions": "Tutoring sessions",
        "Physical_Activity": "Physical activity",
        "Access_to_Resources": "Access to learning resources",
        "Parental_Involvement": "Parental involvement",
        "Internet_Access": "Internet access",
        "Learning_Disabilities": "Reported learning disability status",
        "Peer_Influence": "Peer influence",
        "Motivation_Level": "Motivation level",
        "Family_Income": "Family income category",
        "Teacher_Quality": "Teacher quality",
        "School_Type": "School type",
        "Parental_Education_Level": "Parental education level",
        "Distance_from_Home": "Distance from home to school",
        "Extracurricular_Activities": "Extracurricular activity participation",
        "Gender": "Gender category recorded in the dataset"
    }

    rule = rule.strip()

    for feature in sorted(readable_names, key=len, reverse=True):
        if feature in rule:
            name = readable_names[feature]
            remainder = rule.replace(feature, "", 1).strip()

            if "=" in remainder:
                value = remainder.split("=", 1)[1].strip()
                return f"{name}: {value}."

            if remainder:
                return f"{name} falls within the range {remainder}."

            return name

    return rule


def predict_for_lime(
    values,
    feature_names,
    numerical_features,
    categorical_features,
    category_names,
    preprocessor,
    model,
    numeric_medians,
    categorical_modes
):
    frame = pd.DataFrame(values, columns=feature_names)

    for col in numerical_features:
        frame[col] = pd.to_numeric(
            frame[col],
            errors="coerce"
        ).fillna(numeric_medians[col])

    for col in categorical_features:
        max_code = len(category_names[col]) - 1

        codes = (
            pd.to_numeric(frame[col], errors="coerce")
            .fillna(0)
            .round()
            .clip(0, max_code)
            .astype(int)
        )

        frame[col] = codes.map(category_names[col])

        frame[col] = frame[col].fillna(categorical_modes[col])

    processed = preprocessor.transform(frame[feature_names])

    return model.predict(processed)


try:
    (
        model,
        preprocessor,
        df,
        X,
        X_train,
        feature_names,
        category_maps,
        category_names,
        lime_explainer,
        shap_explainer
    ) = load_project()

except Exception as error:
    st.error("The model, preprocessor or dataset could not be loaded.")
    st.code(str(error))
    st.stop()


st.title("🎓 Student Academic Risk Prediction")

st.write(
    "Estimate academic performance, understand factors influencing "
    "the estimate, and identify practical ways to support learning."
)

st.info(
    "The predicted score is an estimate, not a fixed outcome or a measure "
    "of a student's potential. The suggestions are intended to support "
    "learning, not label or judge a student."
)

st.subheader("Enter Student Details")

with st.form("student_details_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        hours_studied = st.number_input(
            "Hours Studied per Week",
            min_value=0.0,
            max_value=60.0,
            value=20.0
        )

        attendance = st.number_input(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0
        )

        previous_scores = st.number_input(
            "Previous Scores",
            min_value=0.0,
            max_value=100.0,
            value=75.0
        )

        sleep_hours = st.number_input(
            "Sleep Hours per Day",
            min_value=0.0,
            max_value=24.0,
            value=7.0
        )

        tutoring_sessions = st.number_input(
            "Tutoring Sessions",
            min_value=0,
            max_value=20,
            value=2
        )

        physical_activity = st.number_input(
            "Physical Activity (hours per week)",
            min_value=0.0,
            max_value=20.0,
            value=3.0
        )

    with col2:
        parental_involvement = st.selectbox(
            "Parental Involvement",
            ["Low", "Medium", "High"]
        )

        access_to_resources = st.selectbox(
            "Access to Resources",
            ["Low", "Medium", "High"]
        )

        extracurricular = st.selectbox(
            "Extracurricular Activities",
            ["Yes", "No"]
        )

        motivation = st.selectbox(
            "Motivation Level",
            ["Low", "Medium", "High"]
        )

        internet_access = st.selectbox(
            "Internet Access",
            ["Yes", "No"]
        )

        family_income = st.selectbox(
            "Family Income",
            ["Low", "Medium", "High"]
        )

        teacher_quality = st.selectbox(
            "Teacher Quality",
            ["Low", "Medium", "High"]
        )

    with col3:
        school_type = st.selectbox(
            "School Type",
            ["Public", "Private"]
        )

        peer_influence = st.selectbox(
            "Peer Influence",
            ["Positive", "Neutral", "Negative"]
        )

        learning_disabilities = st.selectbox(
            "Learning Disabilities",
            ["No", "Yes"]
        )

        parental_education = st.selectbox(
            "Parental Education Level",
            ["High School", "College", "Postgraduate"]
        )

        distance = st.selectbox(
            "Distance from Home",
            ["Near", "Moderate", "Far"]
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    submitted = st.form_submit_button(
        "Predict Academic Performance",
        use_container_width=True
    )


if submitted:

    student = pd.DataFrame([{
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Parental_Involvement": parental_involvement,
        "Access_to_Resources": access_to_resources,
        "Extracurricular_Activities": extracurricular,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores,
        "Motivation_Level": motivation,
        "Internet_Access": internet_access,
        "Tutoring_Sessions": tutoring_sessions,
        "Family_Income": family_income,
        "Teacher_Quality": teacher_quality,
        "School_Type": school_type,
        "Peer_Influence": peer_influence,
        "Physical_Activity": physical_activity,
        "Learning_Disabilities": learning_disabilities,
        "Parental_Education_Level": parental_education,
        "Distance_from_Home": distance,
        "Gender": gender
    }])[feature_names]

    processed_student = preprocessor.transform(student)
    predicted_score = float(model.predict(processed_student)[0])
    risk = classify_risk(predicted_score)

    st.divider()
    st.subheader("Your Academic Prediction")

    metric1, metric2 = st.columns(2)

    metric1.metric(
        "Previous Examination Score",
        f"{previous_scores:.1f} / 100"
    )

    metric2.metric(
        "Model-Estimated Score",
        f"{predicted_score:.1f} / 100"
    )

    st.write(
        f"**Academic indicator:** {risk}"
    )

    st.caption(
        "This category uses provisional thresholds: below 60 is High Risk, "
        "60 to below 65 is Moderate Risk, and 65 or above is Lower Predicted "
        "Risk. These thresholds have not been validated as a student assessment."
    )

    st.write(
        "A difference between the previous score and this estimate does not "
        "guarantee a future improvement or decline. Use the explanation as "
        "a starting point for supportive academic planning."
    )

    st.divider()
    st.subheader("SHAP — Suggestions for Improvement")

    try:
        transformed = (
            processed_student.toarray()
            if hasattr(processed_student, "toarray")
            else np.asarray(processed_student)
        )

        shap_values = np.asarray(
            shap_explainer.shap_values(transformed)
        ).reshape(-1)

        transformed_names = preprocessor.get_feature_names_out()

        contributions = {}

        for name, value in zip(transformed_names, shap_values):
            clean_name = name

            if clean_name.startswith("numerical__"):
                original_feature = clean_name.replace("numerical__", "", 1)

            elif clean_name.startswith("categorical__"):
                suffix = clean_name.replace("categorical__", "", 1)

                matches = [
                    feature for feature in CATEGORICAL_FEATURES
                    if suffix == feature or suffix.startswith(feature + "_")
                ]

                original_feature = (
                    max(matches, key=len) if matches else suffix
                )

            else:
                original_feature = clean_name

            contributions[original_feature] = (
                contributions.get(original_feature, 0.0) + float(value)
            )

        suggestions = build_suggestions(
            student.iloc[0],
            contributions
        )

        for index, (title, message) in enumerate(suggestions, start=1):
            st.markdown(f"**{index}. {title}**")
            st.write(message)

    except Exception as error:
        st.warning("SHAP suggestions could not be generated.")
        st.caption(str(error))

    st.divider()
    st.subheader("LIME — Five Important Factors")

    try:
        numeric_medians = {
            col: float(X_train[col].median())
            for col in NUMERICAL_FEATURES
        }

        categorical_modes = {
            col: X_train[col].mode().iloc[0]
            for col in CATEGORICAL_FEATURES
        }

        student_lime = student.copy()

        for col in NUMERICAL_FEATURES:
            student_lime[col] = student_lime[col].fillna(
                numeric_medians[col]
            )

        for col in CATEGORICAL_FEATURES:
            student_lime[col] = student_lime[col].map(
                category_maps[col]
            )

        student_lime = student_lime[feature_names].astype(float)

        lime_explanation = lime_explainer.explain_instance(
            data_row=student_lime.iloc[0].to_numpy(),
            predict_fn=lambda values: predict_for_lime(
                values,
                feature_names,
                NUMERICAL_FEATURES,
                CATEGORICAL_FEATURES,
                category_names,
                preprocessor,
                model,
                numeric_medians,
                categorical_modes
            ),
            num_features=5,
            num_samples=3000
        )

        lime_results = pd.DataFrame(
            lime_explanation.as_list(),
            columns=["Feature Rule", "LIME Contribution"]
        )

        lime_results["Absolute Contribution"] = (
            lime_results["LIME Contribution"].abs()
        )

        lime_results = lime_results.sort_values(
            "Absolute Contribution",
            ascending=False
        ).head(5).copy()

        total_contribution = lime_results[
            "Absolute Contribution"
        ].sum()

        if total_contribution > 0:
            lime_results["Relative Contribution (%)"] = (
                lime_results["Absolute Contribution"]
                / total_contribution
                * 100
            )
        else:
            lime_results["Relative Contribution (%)"] = 0.0

        st.write(
            "These five factors show how the model's local explanation "
            "supports a higher or lower estimate for this student."
        )

        for _, row in lime_results.iterrows():
            label = explain_lime_rule(row["Feature Rule"])

            contribution = float(row["LIME Contribution"])
            percentage = float(row["Relative Contribution (%)"])

            if contribution > 0:
                direction = "Pushes the estimate higher"
            elif contribution < 0:
                direction = "Pushes the estimate lower"
            else:
                direction = "Little or no directional contribution"

            st.markdown(f"**{label}**")
            st.write(
                f"{direction} · Relative contribution: {percentage:.1f}%"
            )

            bar_value = min(max(percentage, 0.0), 100.0)

            st.progress(int(round(bar_value)))

        st.caption(
            "Relative contribution percentages are calculated from the "
            "absolute LIME contributions of these five displayed factors. "
            "They are not percentages of marks gained or lost, and do not "
            "measure causal effects. LIME is a local approximation."
        )

    except Exception as error:
        st.warning("LIME explanations could not be generated.")
        st.caption(str(error))

    st.divider()

    st.caption(
        "This tool is intended for educational exploration and supportive "
        "planning. It should not be used alone to make decisions about a "
        "student's future or ability."
    )
