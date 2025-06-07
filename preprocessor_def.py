from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer

std_scale_features = ['F3', 'F4']
power_transform_features = ['F5']

preprocessor = ColumnTransformer(
    transformers=[
        ('std_scaler', StandardScaler(), std_scale_features),
        ('power_transform', PowerTransformer(method='yeo-johnson'), power_transform_features)
    ],
    remainder='passthrough' 
)
