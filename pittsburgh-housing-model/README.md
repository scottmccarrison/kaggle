### Pittsburgh Home Sale Propensity Model
Predict which houses in the Pittsburgh area are most likely to list 
for sale within a given time window (e.g., 6 months).

Binary classification problem using Allegheny County property 
assessment and sales records from the WPRDC. The model scores every 
property that hasn't recently sold, flagging those most likely to 
list next. The strongest known predictor is ownership duration — 
how long the current owner has lived there. Other signals include 
owner type (primary resident vs. absentee/investor), equity position, 
property age and condition, and spatial features (density of nearby 
recent sales, median price of those sales).

This project combines skills from multiple tour projects: binary 
classification (Titanic), class imbalance (Credit Card Fraud), and 
real estate domain knowledge (House Prices). The key challenge is 
data acquisition — joining public county records to build owner 
histories and spatial features, rather than working with a 
pre-packaged Kaggle dataset.

**Prerequisites:** Titanic, Credit Card Fraud, House Prices  
**Data source:** WPRDC — Allegheny County property assessment and 
sales records  
**New skills:** Spatial feature engineering, real-world data 
acquisition and joining, survival analysis, public government datasets  
`Classification` `Binary` `Spatial Data` `Real Estate` `Survival Analysis` `WPRDC` `Pittsburgh`
