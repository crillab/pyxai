# Changelog

### 1.0.10
 - Contrastive for BT classification (binary classes)
 - Remove the PyQt6 dependencie and new methods to display explanations:
    - show_in_notebook()
    - show_on_screen()
    - get_PILImage()
    - save_png()
    - resize_PILimage()
 - Change function name in explainer (unset_specific_features -> unset_excluded_features)
 - New procedure installation
 - New visualization for time series
 - Compilation error resolution
 
### 1.0.9
 - New metrics (documentation in progress)
   - For binary classification:
    - accuracy
    - precision
    - recall
    - f1_score
    - specificity
    - tp, tn, fp, fn
  - For multiclass classification:
    - micro_averaging_accuracy
    - micro_averaging_precision
    - micro_averaging_recall
    - macro_averaging_accuracy
    - macro_averaging_precision
    - macro_averaging_recall
  - For regression:
    - mean_squared_error
    - root_mean_squared_error
    - mean_absolute_error

### 1.0.7
 - Build and Tests with CI

### 1.0.0
 - Regression for boosted trees
 - Adding thoeries
 - Easier import model 
 - Graphical user interface: displaying, loading, saving explanations
 - Data preprocessing
 - unit tests
## 0.X
 - Initial release