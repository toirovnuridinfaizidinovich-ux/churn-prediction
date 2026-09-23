import streamlit as st
import pandas as pd

from predict import predict_batch


st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Churn Prediction")
st.write("Загрузи CSV с клиентами — приложение вернёт таблицу с риском оттока.")


def get_risk(probability):
    if probability >= 0.70:
        return "high"
    elif probability >= 0.40:
        return "medium"
    return "low"


uploaded_file = st.file_uploader(
    "Выбери CSV-файл с клиентами",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Входные данные")
        st.write(f"Строк загружено: {len(df)}")
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("Сделать прогноз", type="primary"):
            with st.spinner("Модель считает риск оттока..."):
                result = predict_batch(df)

                result["risk_segment"] = result["churn_probability"].apply(
                    get_risk
                )

                result = result.sort_values(
                    by="churn_probability",
                    ascending=False
                )

            st.success("Готово")

            high_risk_count = (result["risk_segment"] == "high").sum()
            medium_risk_count = (result["risk_segment"] == "medium").sum()
            low_risk_count = (result["risk_segment"] == "low").sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Высокий риск", int(high_risk_count))
            col2.metric("Средний риск", int(medium_risk_count))
            col3.metric("Низкий риск", int(low_risk_count))

            st.subheader("Результат")
            st.dataframe(result, use_container_width=True)

            csv = result.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                label="Скачать CSV с прогнозами",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

    except Exception as error:
        st.error("Не удалось обработать файл.")
        st.code(str(error))