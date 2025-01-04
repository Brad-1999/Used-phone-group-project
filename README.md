# Used Smartphone E-commerce Analysis

This project analyzes a dataset of used smartphones scraped from Chotot.com, an online marketplace. The project includes data collection, preprocessing, exploratory data analysis, hypothesis testing, machine learning for price prediction, and feature importance analysis.

## Project Structure

The project is organized as follows:

*   **`crawler/`:**
    -   Contains the Scrapy crawler files.
        *   `spiders/`: Includes the spider implementation (`chotot.py`).
        *   `items.py`: Defines the data structure of the scraped data.
        *   `middlewares.py`: Includes the middleware of the Scrapy project
        *   `pipelines.py`: Defines how to save the scraped data into a file.
        *   `settings.py`: Settings file for the Scrapy project.
*   **`data/`:**
    -   Contains the raw scraped data as JSONL file and the processed data in CSV format.
        *   `cleaned_info.csv`: Processed CSV file ready for analysis.
        *   `description.csv`: CSV file containing the descriptions for the ads.
        *  `info.csv`:  Raw data with the info for each ad.
        * `2024-10-24.jsonl`: Example JSONL file containing the scraped data (19482 records).
        * Example record:

```JSON
        {"listing_id": "120443732", "url": "https://www.chotot.com/mua-ban-dien-thoai-quan-cai-rang-can-tho/120443732.htm", "content": {"props": {"pageProps": {"initialState": {"adView": {"adInfo": {"ad": {"ad_id": 162233828, "list_id": 120443732, "list_time": 1729761924000, "state": "accepted", "type": "s", "account_name": "Hu\u1ef3nh Th\u1ecbnh", "phone": "076395****", "region": 5, "category": 5010, "subject": "Nh\u00e0 d\u01b0 m\u00e1y c\u1ea7n b\u00e1n realme 5 pro", "body": "L\u01b0ng b\u1ecb nh\u0103m nh\u0103m c\u00f2n l\u1ea1i b\u00ecnh th\u01b0\u1eddng full ch\u1ee9c n\u0103ng", "price": 800000, "image": "https://cdn.chotot.com/CcLOgPkRBOb3mFvsOegZ7T9P8DR3s6e2QLp7ZyEmn3A/preset:listing/plain/74cedd98937f14d923b397cd0fae6613-2902058751137635396.jpg", "account_id": 18156398, "images": ["https://cdn.chotot.com/8VY-D2ZTXX0w7H0N-tTNHs4jV3-jLoXL5lBAertaj8k/preset:view/plain/74cedd98937f14d923b397cd0fae6613-2902058751137635396.jpg", "https://cdn.chotot.com/6iG_K9OKzHLPPZZJboD_OF2E_t9AjRisxj2TtPWuXbU/preset:view/plain/a1a5d94b8849f0d62523192bf4ffa631-2902058751146938174.jpg"], "videos": [], "contain_videos": 2, "status": "active", "elt_condition": 2, "elt_warranty": 1, "mobile_brand": 34, "mobile_capacity": 6, "mobile_color": 6, "mobile_model": 2055, "area": 27, "longitude": 105.7762, "latitude": 9.994971, "pty_characteristics": [], "region_v2": 5027, "area_v2": 502704, "ward": 10758, "elt_origin": 1, "location": "9.9949715,105.776198", "label_campaigns": [], "escrow_can_deposit": 2, "escrow_can_delivery": true, "inspection_images": [], "gds_inspected": 2, "full_name": "Hu\u1ef3nh Th\u1ecbnh", "total_rating": 1, "average_rating": 5, "date": "13 ph\u00fat tr\u01b0\u1edbc", "account_oid": "01627847280c335014969f6193888f80", "category_name": "\u0110i\u1ec7n tho\u1ea1i", "area_name": "Qu\u1eadn C\u00e1i R\u0103ng", "region_name": "C\u1ea7n Th\u01a1", "price_string": "800.000 \u0111", "webp_image": "https://cdn.chotot.com/4eaprlFDxuY0uiTjy1tq12ipcNqLV163591KKUEVZKs/preset:listing/plain/74cedd98937f14d923b397cd0fae6613-2902058751137635396.webp", "image_thumbnails": [{"image": "https://cdn.chotot.com/8VY-D2ZTXX0w7H0N-tTNHs4jV3-jLoXL5lBAertaj8k/preset:view/plain/74cedd98937f14d923b397cd0fae6613-2902058751137635396.jpg", "thumbnail": "https://cdn.chotot.com/CcLOgPkRBOb3mFvsOegZ7T9P8DR3s6e2QLp7ZyEmn3A/preset:listing/plain/74cedd98937f14d923b397cd0fae6613-2902058751137635396.jpg"}, {"image": "https://cdn.chotot.com/6iG_K9OKzHLPPZZJboD_OF2E_t9AjRisxj2TtPWuXbU/preset:view/plain/a1a5d94b8849f0d62523192bf4ffa631-2902058751146938174.jpg", "thumbnail": "https://cdn.chotot.com/bu5V5Dt8VcEEiVGb3Fi-hxJrLNNCGImS7HhIcA6Mh54/preset:listing/plain/a1a5d94b8849f0d62523192bf4ffa631-2902058751146938174.jpg"}], "special_display_images": [], "number_of_images": 2, "ad_features": [], "avatar": "https://cdn.chotot.com/uac2/18156398", "ward_name": "Ph\u01b0\u1eddng H\u01b0ng Th\u1ea1nh", "thumbnail_image": "https://cdn.chotot.com/CcLOgPkRBOb3mFvsOegZ7T9P8DR3s6e2QLp7ZyEmn3A/preset:listing/plain/74cedd98937f14d923b397cd0fae6613-2902058751137635396.jpg", "params": [{"id": "mobile_model", "value": "5 Pro", "label": "D\u00f2ng m\u00e1y"}, {"id": "mobile_capacity", "value": "128 GB", "label": "Dung l\u01b0\u1ee3ng"}, {"id": "elt_warranty", "value": "H\u1ebft b\u1ea3o h\u00e0nh", "label": "Ch\u00ednh s\u00e1ch b\u1ea3o h\u00e0nh"}], "ad_labels": [], "size_unit_string": "m\u00b2"}, "ad_params": {"address": {"id": "address", "label": "\u0110\u1ecba ch\u1ec9", "value": "Ph\u01b0\u1eddng H\u01b0ng Th\u1ea1nh, Qu\u1eadn C\u00e1i R\u0103ng, C\u1ea7n Th\u01a1"}, "area": {"id": "area", "label": "Qu\u1eadn, Huy\u1ec7n", "value": "Qu\u1eadn C\u00e1i R\u0103ng"}, "elt_condition": {"id": "elt_condition", "label": "T\u00ecnh tr\u1ea1ng", "value": "\u0110\u00e3 s\u1eed d\u1ee5ng (ch\u01b0a s\u1eeda ch\u1eefa)"}, "elt_origin": {"id": "elt_origin", "label": "Xu\u1ea5t x\u1ee9", "value": "Vi\u1ec7t Nam"}, "elt_warranty": {"id": "elt_warranty", "label": "Ch\u00ednh s\u00e1ch b\u1ea3o h\u00e0nh", "value": "H\u1ebft b\u1ea3o h\u00e0nh"}, "mobile_brand": {"id": "mobile_brand", "label": "H\u00e3ng", "value": "Realme"}, "mobile_capacity": {"id": "mobile_capacity", "label": "Dung l\u01b0\u1ee3ng", "value": "128 GB"}, "mobile_color": {"id": "mobile_color", "label": "M\u00e0u s\u1eafc", "value": "Xanh d\u01b0\u01a1ng"}, "mobile_model": {"id": "mobile_model", "label": "D\u00f2ng m\u00e1y", "value": "5 Pro"}, "region": {"id": "region", "label": "T\u1ec9nh, th\u00e0nh ph\u1ed1", "value": "C\u1ea7n Th\u01a1"}, "usage_information": {"id": "usage_information", "label": "Th\u00f4ng tin s\u1eed d\u1ee5ng", "value": "In tr\u00ean bao b\u00ec"}, "ward": {"id": "ward", "label": "Ph\u01b0\u1eddng, th\u1ecb x\u00e3, th\u1ecb tr\u1ea5n", "value": "Ph\u01b0\u1eddng H\u01b0ng Th\u1ea1nh"}}}}, "nav": {"navObj": {"headers": {"content-type": "application/json; charset=utf-8", "content-length": "610", "connection": "close", "vary": "Accept-Encoding, Accept-Encoding, Origin", "x-powered-by": "Express", "etag": "W/\"262-jXcK0wTeMQf3nKkz+82ddsDiXl0\"", "date": "Thu, 24 Oct 2024 09:38:30 GMT", "access-control-expose-headers": "X-Total-Count", "x-kong-upstream-latency": "2", "x-kong-proxy-latency": "1"}, "param1": "mua-ban-dien-thoai-quan-cai-rang-can-tho", "productId": "120443732", "numSeoMatched": 0, "postParamMatched": 0, "region": "can-tho", "subRegion": "quan-cai-rang", "adType": "s,k", "category": "dien-thoai", "paramObj": {"cg": "5010", "region_v2": "5027", "area_v2": "502704", "st": "s,k"}, "categoryObj": {"label": "\u0110i\u1ec7n tho\u1ea1i", "value": 5010, "route": "dien-thoai"}, "queryObj": {}, "regionObj": {}, "regionObjV2": {"regionValue": 5027, "regionUrl": "can-tho", "regionName": "C\u1ea7n Th\u01a1", "subRegionValue": 502704, "subRegionUrl": "quan-cai-rang", "subRegionName": "Qu\u1eadn C\u00e1i R\u0103ng", "empty_region": false}, "is_resolved": true, "success": true}}}}}}, "crawl_date": "2024-10-24", "source": "chotot"}
```

*   **`notebooks/`:**
    -   Contains Jupyter notebooks for data exploration, preprocessing, hypothesis testing, and model analysis.
        *   `data_cleaning_and_eda.ipynb`: Jupyter notebook for data cleaning, EDA, as well as forming some initial hypotheses.
        *   `hypotheses_testing.ipynb`: Jupyter notebook for hypotheses testing.
*   **`src/`:**
    -   Contains the Python files for the project pipeline.
        *   `csv_extraction.py`: Handles the conversion of JSONL files to pandas DataFrames.
        *  `preprocessing.py`: Handles the data preprocessing.
        *  `trainer.py`: Contains the machine learning model training and analysis functions.
        *  `utils.py`: Contains ultility functions to help the training process.
*   **`.gitignore`:** Specifies intentionally untracked files that Git should ignore.
*   **`README.md`:** This file, providing an overview of the project.
*   **`requirements.txt`:** Lists the Python dependencies required to run the project.
*   **`scrapy.cfg`:** Configuration file for Scrapy.
*   **`main.py`:** Script for training and evaluating the models using the full pipeline.

## Data Collection

The data was collected using a Scrapy web crawler that targeted the Chotot.com website. The crawler extracted information about used smartphones, including:

*   Listing details (ID, URL, etc.)
*   Seller information
*   Phone specifications (brand, model, storage capacity, color, condition, etc.)
*   Price and location details
*   Rating information.

The scraped data was saved in JSONL (JSON Lines) format.

## Data Preprocessing

The data preprocessing steps included:

*   **Conversion to CSV:** The JSONL data was converted to a pandas DataFrame using `src/csv_extraction.py`.
*   **Cleaning and Standardization:**
    -   Data cleaning and transformation using `src/preprocessing.py`.
    -   Conversion of prices from VND to USD.
    -   Standardization of categorical features (e.g., condition, origin, warranty, brand, color) using mappings.
    -   Location names and removal of extra characters.
    -   Log transformation of prices to reduce the impact of extreme values.
     * Missing values in `color` are handled by filling with the `unknown` value.

*   **Feature Engineering:**
    <!-- -   Added `color_popularity_score` (numerical score of each color based on frequency). -->

    -   Added `dominant_colors_by_brand` (multi-label categorical variable for the most frequent colors for each brand).
* **Stratified Split**: Added a stratified split to ensure that the datasets have similar distributions (optional).

## Exploratory Data Analysis (EDA)

The data analysis included:
*  **Data type verification**: Checked the data types of the columns using `polars` and `pandas`.
*  **Data distribution**: Verified the distributions of the numerical and categorical variables.
*  **Missing values**: Verified the number of missing values for all columns.
*  **Transformed prices**: Check the effect of log transforming the prices.
*  **Visualization**: All hypotheses were verified using different plots:
    -   Box plots, violin plots, and bar charts for visualizing distributions.
    -   Scatter plots for visualizing correlations.
    -   Heatmaps for visualizing the associations of different variables.
    -   Regression plots, for verifying the goodness of linear regression models.

## Hypothesis Testing

We formulated and tested the following hypotheses:

*I. Price-related Hypotheses:
01.   **Hypothesis 1:** *For each additional GB of storage, the price increases by a statistically significant percentage, and this elasticity differs across brands.*
02.   **Hypothesis 2:** *The mean price of phones sold by companies is significantly higher than the mean price of phones sold by individuals.*
03.   **Hypothesis 3:** *There is a statistically significant difference in prices among phones with different colors.*

*II. Seller/Rating-related Hypotheses:
04.   **Hypothesis 4:** *There is a statistically significant positive correlation between seller ratings and phone prices.*
05.   **Hypothesis 5:** *Company sellers have a statistically significant higher average rating than individual sellers.*
06.   **Hypothesis 6:** *There is a statistically significant positive correlation between the number of `sold_ads` and seller average ratings.*

*III. Geographic-related Hypotheses:
07.   **Hypothesis 7:** *The mean price of phones in major cities (Hanoi, HCMC) is statistically significantly higher than the mean price of phones in other regions.*
08.  **Hypothesis 8:** *There is a statistically significant association between brands and regions.*
09.  **Hypothesis 9:** *The proportion of high-end phones is significantly higher in urban regions compared to rural regions.*

*   **Statistical Tests Used:** We used t-tests, Mann-Whitney U tests, ANOVA, Chi-squared, Pearson correlation, and Spearman correlation based on the hypothesis and normality of the data.
*   **Significance Level:** All tests were conducted at a significance level (alpha) of 0.05.
*   **Multiple Comparisons:** We applied Bonferroni correction to account for multiple comparisons.
*   **Results**: We have detailed the results of each hypothesis, which are supported by the visualizations.

## Machine Learning

### Objective

To predict the price of used smartphones based on their attributes, seller information and geographic details. This will provide insights for sellers, buyers and the platform.

### Features

We used the following features for prediction:
* **Phone Attributes**: brand, model, storage capacity, color, condition.
* **Seller Attributes**: Seller type, ratings, number of sold ads.
* **Geographic Attributes**: Region and urban or rural areas.
* **Engineered Features**: `dominant_colors_by_brand`

### Performance Metrics

*   **Mean Absolute Error (MAE)**: Measures the average absolute difference between predictions and actual values.
*   **Root Mean Squared Error (RMSE)**: Measures the square root of the average squared difference between the predicted and actual values.
*   **R-squared (R2)**: Measures the proportion of variance in the target variable explained by the model.
*   **Mean Absolute Percentage Error (MAPE)**: Measures the prediction error as a percentage of actual values.
*   **Relative Accuracy Count (RAC)**: Measures the number of predictions where the absolute percentage difference between the prediction and the actual value is less than or equal to a specified threshold.

### Models

*   **Baseline**: We used Ridge regression, and we also experimented with k-Nearest Neighbors and Random Forest.
*   **Tuning**: We used a Grid Search to find the optimal values for the hyperparameters in the Random Forest Model.
*   **Validation:** We implemented cross-validation (inside GridSearch) to prevent overfitting.
*   **Feature Importance:** We used SHAP (SHapley Additive exPlanations) to calculate the feature importance for the Random Forest model.
*   **Preprocessing**: We used stratified sampling and log-transformation to address brand-level imbalances and outliers.

### Results

The analysis showed that the Random Forest model performed well in our experiments. SHAP values were also generated for the Random Forest to gain more understanding about the feature importance.

## Libraries Used

* **Scrapy:** Web scraping framework.
* **Polars:** Data manipulation library.
* **Pandas:** Data manipulation and analysis.
* **NumPy:** Numerical computations.
* **Scikit-learn:** Machine learning algorithms and tools.
* **Statsmodels:** Statistical modeling.
* **Plotly:** Interactive visualizations.
* **SHAP:** For calculating the SHAP values.
* **Loguru:** Logging library.
* **Tqdm**: Progress bar library.
* **Unidecode**: For text decoding.

## How to Run the Project

01.  **Clone the repository:**

```bash
    git clone <repository_url>
    cd used-smartphones-ecom
```

02.  **Create and activate a virtual environment:**

```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
```

03.  **Install dependencies:**

```bash
    pip install -r requirements.txt
```

04.  **Run the Scrapy crawler:**

```bash
    cd crawler
    scrapy crawl chotot
```

05.  **Process the scraped data and perform model training:**
Run the `main.py` file by:

```bash
    python src/main.py
```

This will create the CSV from the JSONL file, preprocess the data, and train and evaluate the models.

06.  **Analyze the results in the notebook:**

```bash
    jupyter notebook hypotheses_testing.ipynb
```

## Conclusions and Further Steps

*   **Model Results**: We have found that the random forest model performed well with good R2 and reasonable RMSE and MAE. Further hyperparameter optimization might improve the results.
*   **SHAP Insights**: By looking at the SHAP values, we can have a better understanding of the different contributions of each feature.
*   **Hypothesis Test**: Most of the hypothesis were supported by the data. We found that the prices of Apple are higher, that companies are selling at higher prices, and that cities also have higher prices. We also found a strong association between brands and regions.
*   **Future Improvements:**
    -   Hyperparameter tuning of different models, or exploring new models.
    -   More comprehensive feature engineering.
    -   Add historical price information.
    -   Add more features related to seller performance.
    -   Explore more complex models.

## Authors

*   LE TRAN NHA TRAN - JASMINE (Student ID: 11285100M)
*   DINH VAN LONG - BRAD (Student ID: 11285109M)
