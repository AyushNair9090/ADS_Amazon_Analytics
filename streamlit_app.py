import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import shap


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Amazon Analytics",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("amazon_transactions_100k.csv")


df = load_data()


# ============================================================
# CACHED FILTER / DROPDOWN OPTIONS
# ============================================================

@st.cache_data
def get_filter_options(data):
    return {
        "return_status": sorted(
            data["return_status"].dropna().unique().tolist()
        ),
        "customer_state": sorted(
            data["customer_state"].dropna().unique().tolist()
        ),
        "brand": sorted(
            data["brand"].dropna().unique().tolist()
        ),
        "subcategory": sorted(
            data["subcategory"].dropna().unique().tolist()
        ),
        "festival_name": sorted(
            data["festival_name"].fillna("None").unique().tolist()
        ),
        "payment_method": sorted(
            data["payment_method"].dropna().unique().tolist()
        ),
        "delivery_type": sorted(
            data["delivery_type"].dropna().unique().tolist()
        ),
        "customer_age_group": sorted(
            data["customer_age_group"].dropna().unique().tolist()
        ),
        "customer_city": sorted(
            data["customer_city"].dropna().unique().tolist()
        ),
        "customer_tier": sorted(
            data["customer_tier"].dropna().unique().tolist()
        ),
        "customer_spending_tier": sorted(
            data["customer_spending_tier"].dropna().unique().tolist()
        )
    }


FILTER_OPTIONS = get_filter_options(df)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("models/streamlit_model.pkl")


model = load_model()


# ============================================================
# CACHED DRIFT DATA
# ============================================================

@st.cache_data
def prepare_drift_data(data):

    drift_data = data[
        [
            "order_date",
            "quantity",
            "original_price_inr",
            "discounted_price_inr",
            "discount_percent",
            "final_amount_inr",
            "delivery_days",
            "product_rating",
            "product_weight_kg",
            "customer_rating"
        ]
    ].copy()

    drift_data["order_date"] = pd.to_datetime(
        drift_data["order_date"],
        errors="coerce"
    )

    drift_data = drift_data.dropna(
        subset=["order_date"]
    )

    drift_data = drift_data.sort_values(
        "order_date"
    ).reset_index(drop=True)

    return drift_data


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📊 Amazon Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Model Performance",
        "Explainability",
        "Drift Monitoring",
        "Responsible AI",
        "Prediction"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("📊 Amazon Transactions Analytics")

    st.markdown(
        "Interactive analytics dashboard for exploring Amazon "
        "transaction patterns and order outcomes."
    )

    st.divider()

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    st.sidebar.header("🔎 Dashboard Filters")

    status_options = FILTER_OPTIONS["return_status"]

    selected_status = st.sidebar.multiselect(
        "Return Status",
        options=status_options,
        default=status_options
    )

    state_options = FILTER_OPTIONS["customer_state"]

    selected_states = st.sidebar.multiselect(
        "Customer State",
        options=state_options
    )

    # --------------------------------------------------------
    # FILTER DATA
    # --------------------------------------------------------

    if selected_status:

        status_mask = df["return_status"].isin(
            selected_status
        )

    else:

        status_mask = pd.Series(
            True,
            index=df.index
        )

    if selected_states:

        state_mask = df["customer_state"].isin(
            selected_states
        )

    else:

        state_mask = pd.Series(
            True,
            index=df.index
        )

    filtered_df = df.loc[
        status_mask & state_mask
    ]

    # --------------------------------------------------------
    # ORDER STATUS DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("📦 Order Status Distribution")

    dashboard_df = filtered_df[
        ["return_status"]
    ]

    status_counts = (
        dashboard_df["return_status"]
        .value_counts()
        .reset_index()
    )

    status_counts.columns = [
        "return_status",
        "count"
    ]

    fig_status = px.bar(
        status_counts,
        x="return_status",
        y="count",
        text="count",
        title="Orders by Return Status"
    )

    fig_status.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig_status,
        width="stretch"
    )

    # --------------------------------------------------------
    # FILTER SUMMARY
    # --------------------------------------------------------

    st.divider()

    st.subheader("📋 Filter Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:

        st.metric(
            "Filtered Orders",
            f"{len(filtered_df):,}"
        )

    with summary_col2:

        st.metric(
            "Available Dataset Records",
            f"{len(df):,}"
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title("🤖 Model Performance")

    st.markdown(
        "Performance summary of the machine learning model "
        "developed in Experiment 4 and carried forward to "
        "the deployment workflow."
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.subheader("Selected Model")

    model_col1, model_col2 = st.columns(2)

    with model_col1:

        st.info(
            "Logistic Regression"
        )

        st.write(
            "**Model Type:** Classification"
        )

        st.write(
            "**Target:** return_status"
        )

    with model_col2:

        st.info(
            "Scikit-learn Pipeline"
        )

        st.write(
            "**Pipeline:** Preprocessing + Logistic Regression"
        )

        st.write(
            "**Model Tracking:** DVC + MLflow"
        )

    st.divider()

    # --------------------------------------------------------
    # DOCUMENTED PERFORMANCE
    # --------------------------------------------------------

    st.subheader("📈 Evaluation Result")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    metric_col1.metric(
        "Overall Accuracy",
        "90.67%"
    )

    metric_col2.metric(
        "Target Variable",
        "return_status"
    )

    metric_col3.metric(
        "Evaluation Type",
        "Classification"
    )

    st.caption(
        "The reported accuracy is taken from Experiment 5's "
        "fairness report for the selected model."
    )

    st.divider()

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("📊 Target Class Distribution")

    class_counts = (
        df["return_status"]
        .value_counts()
        .reset_index()
    )

    class_counts.columns = [
        "return_status",
        "count"
    ]

    class_counts["percentage"] = (
        class_counts["count"]
        / class_counts["count"].sum()
        * 100
    )

    fig_classes = px.bar(
        class_counts,
        x="return_status",
        y="count",
        text="percentage",
        title="Return Status Distribution"
    )

    fig_classes.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig_classes,
        width="stretch"
    )

    st.warning(
        "The target classes are imbalanced. Therefore, accuracy "
        "alone should not be treated as a complete assessment "
        "of model performance."
    )

    st.divider()

    # --------------------------------------------------------
    # MODELING WORKFLOW
    # --------------------------------------------------------

    st.subheader("🔄 Modeling Workflow")

    workflow = pd.DataFrame(
        {
            "Stage": [
                "Data Preparation",
                "Baseline Models",
                "Hyperparameter Tuning",
                "Model Comparison",
                "Model Selection",
                "DVC Versioning",
                "API Deployment"
            ],
            "Status": [
                "Completed",
                "Completed",
                "Completed",
                "Completed",
                "Completed",
                "Completed",
                "Completed"
            ]
        }
    )

    st.dataframe(
        workflow,
        hide_index=True,
        width="stretch"
    )

    st.info(
        "Experiment 4 compared multiple classification algorithms "
        "and performed hyperparameter tuning. Experiment 7 then "
        "versioned the selected Logistic Regression pipeline with "
        "DVC and integrated it into the FastAPI deployment workflow."
    )


# ============================================================
# EXPLAINABILITY
# ============================================================

elif page == "Explainability":

    st.header("🔍 Model Explainability")

    st.write(
        "SHAP (SHapley Additive exPlanations) is used to explain "
        "how individual features contribute to the model prediction."
    )

    # --------------------------------------------------------
    # SAMPLE SIZE
    # --------------------------------------------------------

    sample_size = st.slider(
        "Number of records used for explanation",
        min_value=10,
        max_value=50,
        value=20,
        step=10
    )

    X = df.drop(
        columns=["return_status"]
    )

    sample_data = X.head(sample_size)

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    classifier = model.named_steps[
        "model"
    ]

    # --------------------------------------------------------
    # SHAP CALCULATION
    # --------------------------------------------------------

    @st.cache_data
    def calculate_shap_values(
        _model,
        sample_data
    ):

        preprocessor = _model.named_steps[
            "preprocessor"
        ]

        classifier = _model.named_steps[
            "model"
        ]

        X_transformed = preprocessor.transform(
            sample_data
        )

        explainer = shap.LinearExplainer(
            classifier,
            X_transformed
        )

        shap_values = explainer(
            X_transformed
        )

        feature_names = (
            preprocessor.get_feature_names_out()
        )

        return (
            shap_values.values,
            feature_names
        )

    shap_values_array, feature_names = (
        calculate_shap_values(
            model,
            sample_data
        )
    )

    # --------------------------------------------------------
    # GLOBAL FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader("Global Feature Importance")

    mean_abs_shap = (
        abs(shap_values_array)
        .mean(axis=(0, 2))
    )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Mean |SHAP Value|": mean_abs_shap
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "Mean |SHAP Value|",
            ascending=False
        )
        .head(15)
    )

    importance_df = (
        importance_df
        .sort_values("Mean |SHAP Value|")
    )

    st.bar_chart(
        importance_df.set_index(
            "Feature"
        )
    )

    st.caption(
        "Higher mean absolute SHAP values indicate features that "
        "have a greater influence on the model's predictions."
    )

    st.divider()

    # --------------------------------------------------------
    # INDIVIDUAL PREDICTION EXPLANATION
    # --------------------------------------------------------

    st.subheader(
        "Individual Prediction Explanation"
    )

    record_index = st.number_input(
        "Select record index",
        min_value=0,
        max_value=sample_size - 1,
        value=0,
        step=1
    )

    selected_record = sample_data.iloc[
        [record_index]
    ]

    prediction = model.predict(
        selected_record
    )[0]

    probabilities = model.predict_proba(
        selected_record
    )[0]

    classes = classifier.classes_

    st.info(
        f"Predicted Return Status: **{prediction}**"
    )

    probability_df = pd.DataFrame(
        {
            "Return Status": classes,
            "Probability": probabilities * 100
        }
    )

    probability_df["Probability"] = (
        probability_df["Probability"]
        .round(2)
    )

    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # LOCAL SHAP
    # --------------------------------------------------------

    selected_shap = (
        shap_values_array[record_index]
    )

    predicted_class_index = list(
        classes
    ).index(prediction)

    selected_shap_class = (
        selected_shap[
            :,
            predicted_class_index
        ]
    )

    local_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "SHAP Value": selected_shap_class
        }
    )

    local_df["Absolute SHAP"] = (
        local_df["SHAP Value"].abs()
    )

    local_df = (
        local_df
        .sort_values(
            "Absolute SHAP",
            ascending=False
        )
        .head(15)
    )

    local_df = local_df.sort_values(
        "SHAP Value"
    )

    st.subheader(
        f"Top Factors Influencing '{prediction}' Prediction"
    )

    st.bar_chart(
        local_df.set_index(
            "Feature"
        )["SHAP Value"]
    )

    st.caption(
        "Positive SHAP values push the prediction toward the "
        "selected class, while negative values push it away "
        "from that class."
    )

    st.divider()

    # --------------------------------------------------------
    # EXPLAINABILITY METHODS
    # --------------------------------------------------------

    st.subheader(
        "Explainability Methods Used"
    )

    method_df = pd.DataFrame(
        {
            "Method": [
                "SHAP",
                "LIME"
            ],
            "Purpose": [
                "Global and local feature contribution analysis",
                "Local, instance-level model explanation"
            ]
        }
    )

    st.dataframe(
        method_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DRIFT MONITORING
# ============================================================

elif page == "Drift Monitoring":

    st.header("📊 Data Drift Monitoring")

    st.write(
        "This module compares an earlier reference period with a later "
        "current period to identify changes in the distribution of key "
        "numerical features."
    )

    # Drift data is prepared only when this page is opened.

    drift_df = prepare_drift_data(df)

    if len(drift_df) < 20:

        st.warning(
            "Not enough valid dated records for drift analysis."
        )

    else:

        # ----------------------------------------------------
        # REFERENCE / CURRENT PERIOD
        # ----------------------------------------------------

        split_index = int(
            len(drift_df) * 0.8
        )

        reference_df = drift_df.iloc[
            :split_index
        ]

        current_df = drift_df.iloc[
            split_index:
        ]

        reference_start = (
            reference_df["order_date"]
            .min()
            .date()
        )

        reference_end = (
            reference_df["order_date"]
            .max()
            .date()
        )

        current_start = (
            current_df["order_date"]
            .min()
            .date()
        )

        current_end = (
            current_df["order_date"]
            .max()
            .date()
        )

        # ----------------------------------------------------
        # MONITORING PERIODS
        # ----------------------------------------------------

        st.subheader("Monitoring Periods")

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Reference Records",
                f"{len(reference_df):,}"
            )

            st.caption(
                f"{reference_start} → {reference_end}"
            )

        with c2:

            st.metric(
                "Current Records",
                f"{len(current_df):,}"
            )

            st.caption(
                f"{current_start} → {current_end}"
            )

        st.info(
            "The reference period contains the earliest 80% of dated "
            "records, while the current period contains the latest 20%. "
            "This is an internal dataset drift analysis, not a live "
            "production monitoring feed."
        )

        st.divider()

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        numerical_features = [
            "quantity",
            "original_price_inr",
            "discounted_price_inr",
            "discount_percent",
            "final_amount_inr",
            "delivery_days",
            "product_rating",
            "product_weight_kg",
            "customer_rating"
        ]

        available_features = [
            col
            for col in numerical_features
            if col in drift_df.columns
        ]

        selected_feature = st.selectbox(
            "Select numerical feature",
            available_features
        )

        # ----------------------------------------------------
        # CONVERT VALUES
        # ----------------------------------------------------

        reference_values = pd.to_numeric(
            reference_df[selected_feature],
            errors="coerce"
        ).dropna()

        current_values = pd.to_numeric(
            current_df[selected_feature],
            errors="coerce"
        ).dropna()

        if (
            len(reference_values) == 0
            or len(current_values) == 0
        ):

            st.warning(
                "Insufficient numeric data available for this feature."
            )

        else:

            # ------------------------------------------------
            # STATISTICS
            # ------------------------------------------------

            reference_mean = (
                reference_values.mean()
            )

            current_mean = (
                current_values.mean()
            )

            reference_median = (
                reference_values.median()
            )

            current_median = (
                current_values.median()
            )

            reference_std = (
                reference_values.std()
            )

            current_std = (
                current_values.std()
            )

            if reference_mean != 0:

                mean_change = (
                    (
                        current_mean
                        - reference_mean
                    )
                    / abs(reference_mean)
                ) * 100

            else:

                mean_change = 0

            # ------------------------------------------------
            # SUMMARY METRICS
            # ------------------------------------------------

            st.subheader(
                f"Distribution Comparison: {selected_feature}"
            )

            m1, m2, m3, m4 = st.columns(4)

            with m1:

                st.metric(
                    "Reference Mean",
                    f"{reference_mean:,.2f}"
                )

            with m2:

                st.metric(
                    "Current Mean",
                    f"{current_mean:,.2f}",
                    f"{mean_change:+.2f}%"
                )

            with m3:

                st.metric(
                    "Reference Median",
                    f"{reference_median:,.2f}"
                )

            with m4:

                st.metric(
                    "Current Median",
                    f"{current_median:,.2f}"
                )

            # ------------------------------------------------
            # LIGHTWEIGHT DISTRIBUTION COMPARISON
            # ------------------------------------------------

            st.subheader(
                "Distribution Comparison"
            )

            combined_min = min(
                reference_values.min(),
                current_values.min()
            )

            combined_max = max(
                reference_values.max(),
                current_values.max()
            )

            if combined_min == combined_max:

                st.info(
                    "The selected feature has the same value "
                    "throughout the compared periods."
                )

            else:

                # Only 30 bins are rendered.

                bins = pd.interval_range(
                    start=combined_min,
                    end=combined_max,
                    periods=30,
                    closed="left"
                )

                reference_bins = pd.cut(
                    reference_values,
                    bins=bins
                ).value_counts(
                    sort=False
                )

                current_bins = pd.cut(
                    current_values,
                    bins=bins
                ).value_counts(
                    sort=False
                )

                distribution_df = pd.DataFrame(
                    {
                        "Reference Period": (
                            reference_bins.values
                        ),
                        "Current Period": (
                            current_bins.values
                        )
                    },
                    index=[
                        str(interval)
                        for interval in bins
                    ]
                )

                # Convert counts to percentages

                distribution_df[
                    "Reference Period"
                ] = (
                    distribution_df[
                        "Reference Period"
                    ]
                    / len(reference_values)
                    * 100
                )

                distribution_df[
                    "Current Period"
                ] = (
                    distribution_df[
                        "Current Period"
                    ]
                    / len(current_values)
                    * 100
                )

                # Reset index for Plotly

                distribution_plot_df = (
                    distribution_df
                    .reset_index()
                    .rename(
                        columns={
                            "index": "Value Range"
                        }
                    )
                )

                # Melt to long format

                distribution_long = (
                    distribution_plot_df
                    .melt(
                        id_vars="Value Range",
                        var_name="Period",
                        value_name="Percentage"
                    )
                )

                fig_distribution = px.bar(
                    distribution_long,
                    x="Value Range",
                    y="Percentage",
                    color="Period",
                    barmode="group",
                    title=(
                        f"Reference vs Current Distribution - "
                        f"{selected_feature}"
                    )
                )

                fig_distribution.update_layout(
                    xaxis_title="Value Range",
                    yaxis_title="Percentage of Records",
                    xaxis_tickangle=-45
                )

                st.plotly_chart(
                    fig_distribution,
                    width="stretch"
                )

            # ------------------------------------------------
            # DISTRIBUTION STATISTICS
            # ------------------------------------------------

            st.subheader(
                "Distribution Statistics"
            )

            statistics_df = pd.DataFrame(
                {
                    "Statistic": [
                        "Mean",
                        "Median",
                        "Standard Deviation",
                        "Minimum",
                        "Maximum",
                        "Record Count"
                    ],
                    "Reference Period": [
                        reference_mean,
                        reference_median,
                        reference_std,
                        reference_values.min(),
                        reference_values.max(),
                        len(reference_values)
                    ],
                    "Current Period": [
                        current_mean,
                        current_median,
                        current_std,
                        current_values.min(),
                        current_values.max(),
                        len(current_values)
                    ]
                }
            )

            statistics_df[
                "Reference Period"
            ] = statistics_df[
                "Reference Period"
            ].round(2)

            statistics_df[
                "Current Period"
            ] = statistics_df[
                "Current Period"
            ].round(2)

            st.dataframe(
                statistics_df,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # DRIFT INTERPRETATION
            # ------------------------------------------------

            st.subheader(
                "Drift Interpretation"
            )

            absolute_change = abs(
                mean_change
            )

            if absolute_change < 5:

                st.success(
                    f"The mean of **{selected_feature}** changed by "
                    f"{absolute_change:.2f}%, indicating relatively low "
                    "distribution change between the two periods."
                )

            elif absolute_change < 15:

                st.warning(
                    f"The mean of **{selected_feature}** changed by "
                    f"{absolute_change:.2f}%, indicating a moderate "
                    "distribution change that should be monitored."
                )

            else:

                st.error(
                    f"The mean of **{selected_feature}** changed by "
                    f"{absolute_change:.2f}%, indicating a substantial "
                    "distribution change that warrants investigation."
                )

            st.caption(
                "The percentage change shown above is a monitoring "
                "indicator, not a formal statistical drift test. "
                "Production monitoring should additionally use "
                "statistical tests such as KS test or PSI and "
                "predefined alert thresholds."
            )

            st.divider()

            # ------------------------------------------------
            # DATA QUALITY
            # ------------------------------------------------

            st.subheader(
                "Data Quality Checks"
            )

            quality_df = pd.DataFrame(
                {
                    "Check": [
                        "Total records",
                        "Valid order dates",
                        "Missing values",
                        "Reference period",
                        "Current period"
                    ],
                    "Result": [
                        f"{len(df):,}",
                        f"{len(drift_df):,}",
                        f"{df.isna().sum().sum():,}",
                        (
                            f"{reference_start} "
                            f"→ {reference_end}"
                        ),
                        (
                            f"{current_start} "
                            f"→ {current_end}"
                        )
                    ]
                }
            )

            st.dataframe(
                quality_df,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# RESPONSIBLE AI
# ============================================================

elif page == "Responsible AI":

    st.title("🛡️ Responsible AI")

    st.markdown(
        "Responsible AI considerations for the Amazon "
        "return-status prediction system."
    )

    # --------------------------------------------------------
    # FAIRNESS
    # --------------------------------------------------------

    st.subheader("Fairness")

    st.write(
        "Experiment 5 evaluated fairness across customer age "
        "groups using Fairlearn."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Demographic Parity Difference",
        "0.0"
    )

    col2.metric(
        "Equalized Odds Difference",
        "0.0"
    )

    col3.metric(
        "Maximum Accuracy Difference",
        "0.013"
    )

    st.caption(
        "These values are from the fairness audit documented "
        "in Experiment 5."
    )

    st.divider()

    # --------------------------------------------------------
    # PRIVACY
    # --------------------------------------------------------

    st.subheader(
        "Privacy"
    )

    st.write(
        "Customer-related information should be handled only "
        "for legitimate analytical purposes. Personally "
        "identifiable information should not be unnecessarily "
        "exposed through the dashboard."
    )

    # --------------------------------------------------------
    # TRANSPARENCY
    # --------------------------------------------------------

    st.subheader(
        "Transparency"
    )

    st.write(
        "The system provides model performance information and "
        "uses SHAP/LIME-based explainability techniques to "
        "support interpretation of predictions."
    )

    # --------------------------------------------------------
    # HUMAN OVERSIGHT
    # --------------------------------------------------------

    st.subheader(
        "Human Oversight"
    )

    st.write(
        "Predictions should be treated as decision-support "
        "outputs rather than unquestionable decisions."
    )

    st.warning(
        "Fairness results apply to the evaluated dataset and "
        "metrics used in Experiment 5. They should not be "
        "interpreted as proof that the model is unbiased under "
        "all future data or deployment conditions."
    )


# ============================================================
# PREDICTION
# ============================================================

elif page == "Prediction":

    st.header(
        "🔮 Return Status Prediction"
    )

    st.write(
        "Enter the order and customer details below to predict "
        "the return status."
    )

    st.subheader(
        "Order Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        order_year = st.number_input(
            "Order Year",
            min_value=2020,
            max_value=2030,
            value=2025
        )

        order_month = st.number_input(
            "Order Month",
            min_value=1,
            max_value=12,
            value=1
        )

        order_quarter = st.number_input(
            "Order Quarter",
            min_value=1,
            max_value=4,
            value=1
        )

        quantity = st.number_input(
            "Quantity",
            min_value=1,
            value=1
        )

        original_price_inr = st.number_input(
            "Original Price (INR)",
            min_value=0.0,
            value=1000.0
        )

    with col2:

        discounted_price_inr = st.number_input(
            "Discounted Price (INR)",
            min_value=0.0,
            value=900.0
        )

        discount_percent = st.number_input(
            "Discount Percent",
            min_value=0.0,
            max_value=100.0,
            value=10.0
        )

        final_amount_inr = st.number_input(
            "Final Amount (INR)",
            min_value=0.0,
            value=900.0
        )

        price_outlier_3sigma = st.selectbox(
            "Price Outlier (3 Sigma)",
            [0, 1]
        )

        price_outlier_IQR = st.selectbox(
            "Price Outlier (IQR)",
            [0, 1]
        )

    with col3:

        is_festival_sale = st.selectbox(
            "Festival Sale",
            [0, 1]
        )

        is_prime_member = st.selectbox(
            "Prime Member",
            [0, 1]
        )

        is_prime_eligible = st.selectbox(
            "Prime Eligible",
            [0, 1]
        )

        delivery_days = st.number_input(
            "Delivery Days",
            min_value=0,
            value=3
        )

        product_rating = st.number_input(
            "Product Rating",
            min_value=0.0,
            max_value=5.0,
            value=4.0
        )

    st.subheader(
        "Product & Customer Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        brand = st.selectbox(
            "Brand",
            FILTER_OPTIONS["brand"]
        )

        subcategory = st.selectbox(
            "Subcategory",
            FILTER_OPTIONS["subcategory"]
        )

        festival_name = st.selectbox(
            "Festival Name",
            FILTER_OPTIONS["festival_name"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            FILTER_OPTIONS["payment_method"]
        )

    with col2:

        delivery_type = st.selectbox(
            "Delivery Type",
            FILTER_OPTIONS["delivery_type"]
        )

        customer_age_group = st.selectbox(
            "Customer Age Group",
            FILTER_OPTIONS["customer_age_group"]
        )

        customer_city = st.selectbox(
            "Customer City",
            FILTER_OPTIONS["customer_city"]
        )

        customer_state = st.selectbox(
            "Customer State",
            FILTER_OPTIONS["customer_state"]
        )

    with col3:

        customer_tier = st.selectbox(
            "Customer Tier",
            FILTER_OPTIONS["customer_tier"]
        )

        customer_spending_tier = st.selectbox(
            "Customer Spending Tier",
            FILTER_OPTIONS[
                "customer_spending_tier"
            ]
        )

        product_weight_kg = st.number_input(
            "Product Weight (kg)",
            min_value=0.0,
            value=1.0
        )

        customer_rating = st.number_input(
            "Customer Rating",
            min_value=0.0,
            max_value=5.0,
            value=4.0
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if st.button(
        "🚀 Predict Return Status",
        type="primary"
    ):

        input_data = pd.DataFrame(
            [
                {
                    "order_year": order_year,
                    "order_month": order_month,
                    "order_quarter": order_quarter,
                    "brand": brand,
                    "subcategory": subcategory,
                    "quantity": quantity,
                    "original_price_inr": original_price_inr,
                    "discounted_price_inr": discounted_price_inr,
                    "discount_percent": discount_percent,
                    "final_amount_inr": final_amount_inr,
                    "price_outlier_3sigma": price_outlier_3sigma,
                    "price_outlier_IQR": price_outlier_IQR,
                    "is_festival_sale": is_festival_sale,
                    "festival_name": festival_name,
                    "is_prime_member": is_prime_member,
                    "is_prime_eligible": is_prime_eligible,
                    "payment_method": payment_method,
                    "delivery_type": delivery_type,
                    "delivery_days": delivery_days,
                    "product_rating": product_rating,
                    "product_weight_kg": product_weight_kg,
                    "customer_age_group": customer_age_group,
                    "customer_city": customer_city,
                    "customer_state": customer_state,
                    "customer_tier": customer_tier,
                    "customer_spending_tier": customer_spending_tier,
                    "customer_rating": customer_rating
                }
            ]
        )

        prediction = model.predict(
            input_data
        )[0]

        probabilities = model.predict_proba(
            input_data
        )[0]

        classes = model.named_steps[
            "model"
        ].classes_

        st.success(
            f"### Predicted Return Status: **{prediction}**"
        )

        st.subheader(
            "Prediction Probabilities"
        )

        probability_df = pd.DataFrame(
            {
                "Return Status": classes,
                "Probability": probabilities
            }
        )

        probability_df[
            "Probability"
        ] = (
            probability_df[
                "Probability"
            ] * 100
        ).round(2)

        st.bar_chart(
            probability_df.set_index(
                "Return Status"
            )["Probability"]
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )