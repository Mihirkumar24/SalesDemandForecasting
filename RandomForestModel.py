import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle

# Load the Excel file
df = pd.read_excel("PreviousSalesRecords.xlsx")

# Clean column names (remove extra spaces, standardize case)
df.columns = df.columns.str.strip()

# Print columns to debug
# print("📋 Columns in your data:", df.columns.tolist())

# Ensure required columns exist
required_columns = ['ACCOUNTNAME', 'PRODUCTID', 'SHIPDATE', 'QUANTITY']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in Excel. Check spelling/case.")

# Convert SHIPDATE to datetime
df['SHIPDATE'] = pd.to_datetime(df['SHIPDATE'], errors='coerce')
df.dropna(subset=['SHIPDATE'], inplace=True)

# Extract Month and Year
df['MONTH'] = df['SHIPDATE'].dt.month
df['YEAR'] = df['SHIPDATE'].dt.year

# Encode categorical features
account_encoder = LabelEncoder()
product_encoder = LabelEncoder()

df['AccountCode'] = account_encoder.fit_transform(df['ACCOUNTNAME'])
df['ProductCode'] = product_encoder.fit_transform(df['PRODUCTID'])

# Prepare feature and target
X = df[['AccountCode', 'ProductCode', 'MONTH', 'YEAR']]
y = df['QUANTITY']

# Split for evaluation (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)   
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"\n📊 Evaluation Metrics:")
print(f"MAE: {mae:.2f}")
#print(f"MAPE: {mape:.2f}%")

# -------------------------------
# 🔍 Prediction by User Input
# -------------------------------
print("\n🔮 Predict Quantity")

input_account = input("Enter Account Name: ").strip()
input_product = input("Enter Product ID: ").strip()
input_month = int(input("Enter Month (1-12): ").strip())
input_year = int(input("Enter Year (e.g., 2025): ").strip())

# Encode inputs
if input_account not in account_encoder.classes_:
    raise ValueError(f"Account '{input_account}' not found in training data.")

if input_product not in product_encoder.classes_:
    raise ValueError(f"Product '{input_product}' not found in training data.")

account_encoded = account_encoder.transform([input_account])[0]
product_encoded = product_encoder.transform([input_product])[0]

# Predict
input_features = pd.DataFrame([[account_encoded, product_encoded, input_month, input_year]],
                              columns=['AccountCode', 'ProductCode', 'MONTH', 'YEAR'])

predicted_quantity = model.predict(input_features)[0]
print(f"\n✅ Predicted Quantity for {input_account} buying {input_product} in {input_month}/{input_year}: {round(predicted_quantity)} units")

# Save the model and encoders
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('account_encoder.pkl', 'wb') as f:
    pickle.dump(account_encoder, f)
with open('product_encoder.pkl', 'wb') as f:
    pickle.dump(product_encoder, f)











'''import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the data
df = pd.read_excel("PreviousSalesRecords.xlsx")

# Clean column names
df.columns = df.columns.str.strip()

# Ensure required columns
required_columns = ['ACCOUNTNAME', 'PRODUCTID', 'SHIPDATE', 'QUANTITY']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' is missing!")

# Convert date and extract features
df['SHIPDATE'] = pd.to_datetime(df['SHIPDATE'], errors='coerce')
df.dropna(subset=['SHIPDATE'], inplace=True)
df['MONTH'] = df['SHIPDATE'].dt.month
df['YEAR'] = df['SHIPDATE'].dt.year

# Unique combinations
combinations = df[['ACCOUNTNAME', 'PRODUCTID']].drop_duplicates()

# Output storage
all_results = []

# Directory to save models
os.makedirs("models", exist_ok=True)

# Loop through each (AccountName, ProductID)
for idx, row in combinations.iterrows():
    account = row['ACCOUNTNAME']
    product = row['PRODUCTID']

    subset = df[(df['ACCOUNTNAME'] == account) & (df['PRODUCTID'] == product)]

    # Skip if data is too small
    if len(subset) < 6:
        continue

    # Features and target
    X = subset[['MONTH', 'YEAR']]
    y = subset['QUANTITY']

    # Split 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions on test set
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Forecast next 2 years (all 12 months)
    future_predictions = []
    last_year = df['YEAR'].max()
    for yr in [last_year + 1, last_year + 7]:
        for mo in range(1, 13):
            pred = model.predict([[mo, yr]])[0]
            future_predictions.append({
                "ACCOUNTNAME": account,
                "PRODUCTID": product,
                "MONTH": mo,
                "YEAR": yr,
                "PREDICTED_QUANTITY": round(pred)
            })

    # Save model
    model_name = f"models/{account}_{product}_model.pkl".replace(" ", "_")
    with open(model_name, "wb") as f:
        pickle.dump(model, f)

    # Store evaluation + predictions
    all_results.append({
        "ACCOUNTNAME": account,
        "PRODUCTID": product,
        "MAE": round(mae, 2),
        "MAPE (%)": round(mape, 2),
        "FUTURE_FORECAST": pd.DataFrame(future_predictions)
    })

# Combine results
metrics_df = pd.DataFrame([{
    "ACCOUNTNAME": r["ACCOUNTNAME"],
    "PRODUCTID": r["PRODUCTID"],
    "MAE": r["MAE"],
    "MAPE (%)": r["MAPE (%)"]
} for r in all_results])

future_df = pd.concat([r["FUTURE_FORECAST"] for r in all_results], ignore_index=True)

# Save to Excel
with pd.ExcelWriter("SalesForecastResults.xlsx") as writer:
    metrics_df.to_excel(writer, sheet_name="Model Metrics", index=False)
    future_df.to_excel(writer, sheet_name="Forecast Next 2 Years", index=False)

print("✅ All models trained, predictions made, and saved to Excel + .pkl files.")'''

