import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bank Customer Classification", layout="wide")
st.title("\U0001F4C8 Bank Customer Classification using CART")

uploaded_file = st.file_uploader("Upload your bank-data.csv file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.dataframe(data.head())

    if 'id' in data.columns:
        data = data.drop("id", axis=1)

    categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
# Biểu đồ 1: Phân phối mục tiêu PEP
    st.subheader("Distribution of Target Variable (PEP)")
    fig_target, ax_target = plt.subplots()
    sns.countplot(x="pep", data=data, ax=ax_target)
    ax_target.set_xticklabels(["No", "Yes"])
    ax_target.set_title("PEP Distribution")
    st.pyplot(fig_target)

    # Biểu đồ 2: Ma trận tương quan giữa các đặc trưng
    st.subheader("Feature Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

    # Biểu đồ 3: Boxplot độ tuổi theo PEP
    st.subheader("Boxplot of Age by PEP")
    fig_box, ax_box = plt.subplots()
    sns.boxplot(x="pep", y="age", data=data, ax=ax_box)
    ax_box.set_xticklabels(["No", "Yes"])
    st.pyplot(fig_box)

    # Biểu đồ 4: Phân phối PEP theo nghề nghiệp
    st.subheader("PEP Distribution by Occupation")
    if "occupation" in data.columns:
        fig_occ, ax_occ = plt.subplots(figsize=(12,6))
        sns.countplot(x="occupation", hue="pep", data=data, ax=ax_occ)
        ax_occ.set_title("PEP Distribution by Occupation")
        plt.xticks(rotation=45)
        st.pyplot(fig_occ)
    if "pep" not in data.columns:
        st.error("The dataset must contain a 'pep' target column.")
    else:
        X = data.drop("pep", axis=1)
        y = data["pep"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

         # Huấn luyện mô hình với tham số chuẩn
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=7,
        max_features=None,
        min_samples_leaf=5,
        min_samples_split=2,
        splitter='best',
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Hiển thị đánh giá mô hình
    st.subheader("Model Evaluation")

    acc = accuracy_score(y_test, y_pred)
    st.write("**Accuracy:**", acc)

    report_str = classification_report(y_test, y_pred)
    st.text("Classification Report:")
    st.code(report_str, language='text')

    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

        st.subheader("Decision Tree Visualization")
        fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
        plot_tree(model, feature_names=X.columns, class_names=["No", "Yes"], filled=True, ax=ax_tree)
        st.pyplot(fig_tree)

        st.subheader("Try Prediction with New Data")
        input_data = {}
        int_columns = ["age", "children"]

        for col in X.columns:
            if col in categorical_cols:
                le = label_encoders[col]
                categories = le.classes_
                selected = st.selectbox(f"{col}", options=categories)
                input_data[col] = le.transform([selected])[0]
            elif col in int_columns:
                val = st.number_input(f"{col}", value=int(X[col].mean()), step=1, format="%d")
                input_data[col] = int(val)
            else:
                val = st.number_input(f"{col}", value=float(X[col].mean()))
                input_data[col] = float(val)

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]

            if prediction == 1:
                st.success("Predicted class: **Yes**")
                st.markdown("\U0001F50D **Khách hàng này có khả năng cao sẽ mua sản phẩm đầu tư PEP.** Bạn nên ưu tiên tiếp cận với các ưu đãi phù hợp.")
            else:
                st.success("Predicted class: **No**")
                st.markdown("ℹ️ **Khách hàng này có khả năng thấp sẽ mua PEP.** Bạn có thể cần chiến lược khác hoặc ưu tiên nhóm khách hàng khác.")
