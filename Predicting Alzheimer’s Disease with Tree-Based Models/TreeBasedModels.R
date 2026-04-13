# ====================================================================
# 1. Data Loading and Preprocessing
# ====================================================================

# Load required libraries
library(dplyr)
library(ggplot2)
library(caret)       
library(rpart)
library(rpart.plot)
library(randomForest)
library(xgboost)
library(lightgbm)
library(pROC)
library(SHAPforxgboost)
library(lime)


# 1. Data Loading
data <- read.csv("alzheimers_disease_data.csv")  # assuming the CSV is in working directory

# Quick overview of the data
#str(data)           # inspect structure (types of columns)
#summary(data)       # summary statistics for numeric features

# 2. Data Cleaning
# Remove unique ID and constant columns that do not carry predictive information
drop.cols <- c("PatientID", "DoctorInCharge")
data <- data %>% dplyr::select(-all_of(drop.cols))

# (PatientID is just an identifier, DoctorInCharge is a constant placeholder "XXXConfid")

# Check for missing values
sum(is.na(data))  # if any column has NA, this will show the count (expected all zeros for this dataset)

# 3. Encoding categorical features
# Convert numeric codes to factors for categorical variables
data$Gender       <- factor(data$Gender, levels = c(0,1), labels = c("Male","Female"))
data$Ethnicity    <- factor(data$Ethnicity, levels = c(0,1,2,3), 
                            labels = c("Caucasian","AfricanAmerican","Asian","Other"))
data$EducationLevel <- factor(data$EducationLevel, levels = c(0,1,2,3),
                              labels = c("None","HighSchool","Undergrad","Higher"))
# Binary Yes/No features (0/1) - convert to factor with descriptive labels
binary_cols <- c("Smoking","FamilyHistoryAlzheimers","CardiovascularDisease",
                 "Diabetes","Depression","HeadInjury","Hypertension",
                 "MemoryComplaints","BehavioralProblems","Confusion",
                 "Disorientation","PersonalityChanges","DifficultyCompletingTasks",
                 "Forgetfulness")
data[binary_cols] <- lapply(data[binary_cols], function(x) factor(x, levels=c(0,1), labels=c("No","Yes")))

# Target variable (Diagnosis: 0 = No AD, 1 = AD) - convert to factor and label
data$Diagnosis <- factor(data$Diagnosis, levels = c(1,0), labels = c("AD","NoAD"))
# (We set "AD" as the first level so it is treated as the positive class in modeling)

# Confirm factor conversions
str(data)  # should show factor types for the above columns


# ====================================================================
# 2. Stratified Train-Test Split
# ====================================================================
# Split data into training (70%) and testing (30%) sets, stratified by Diagnosis
train_index <- createDataPartition(data$Diagnosis, p = 0.7, list = FALSE)
train <- data[train_index, ]
test  <- data[-train_index, ]
cat("Train set size:", nrow(train), "observations\n")
cat("Test set size:", nrow(test), "observations\n")
cat("Train class distribution (%):", round(prop.table(table(train$Diagnosis))*100, 2), "\n")
cat("Test class distribution (%):",  round(prop.table(table(test$Diagnosis))*100, 2), "\n")

# Verify class distribution in split
prop.table(table(train$Diagnosis))  # class proportion in training set
prop.table(table(test$Diagnosis)) 

# ====================================================================
# 3. Model Training with 5-fold Cross-Validation
# ====================================================================

# Classical Tree-Based Models
# Decision Tree (CART)

# Set up cross-validation controls for model training (5-fold CV)
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE,
                     summaryFunction = twoClassSummary)  # using ROC summary

# Train a CART decision tree with cross-validation to tune complexity (cp)
set.seed(123)
model_tree <- train(Diagnosis ~ ., data = train, method = "rpart",
                    trControl = ctrl, metric = "ROC",   # optimize by AUC (ROC)
                    tuneLength = 10)                   # try up to 10 values of cp

# Best tuning parameter
model_tree$bestTune  # optimal cp chosen based on CV ROC

# Inspect the final tree model
model_tree$finalModel  # rpart model details
#printcp(model_tree$finalModel)  # cross-validation results and optimal cp

# Plot the decision tree
rpart.plot(model_tree$finalModel, type = 2, extra = 106, fallen.leaves = TRUE,
           main = "Decision Tree (CART) for Alzheimer’s Diagnosis")

# Random Forest

# Train a Random Forest with 5-fold CV for evaluating performance
set.seed(123)
model_rf <- train(Diagnosis ~ ., data = train, method = "rf",
                  trControl = ctrl, metric = "ROC",  # optimize by AUC
                  tuneLength = 5)   # try a range of mtry values

# Best mtry value
model_rf$bestTune

# Inspect the final random forest model
model_rf$finalModel  # randomForest object summary


# Variable importance from Random Forest
varImp(model_rf)

# Extract the variable importance data.frame
imp_df <- varImp(model_rf)$importance
imp_df$Variable <- rownames(imp_df)

# Keeping only want the top 20 (as printed), you can slice:
imp_df <- imp_df[order(imp_df$Overall, decreasing = TRUE), ][1:20, ]

# Plot as a horizontal bar chart
ggplot(imp_df, aes(x = reorder(Variable, Overall), y = Overall)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Top 20 Variable Importances (Random Forest)",
    x     = NULL,
    y     = "Importance"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.text.y = element_text(face = "italic")
  )

################################ Advanced Gradient Boosting Models

# XGBoost

# XGBoost model training and tuning


set.seed(123)
model_xgb <- train(Diagnosis ~ ., data = train, method = "xgbTree",
                   trControl = ctrl, metric = "ROC", 
                   tuneLength = 5, verbosity=0)  # tune depth, nrounds, etc., with 5 possible combos

# Best hyperparameters
model_xgb$bestTune

# View the XGBoost model details
model_xgb$finalModel

# Extract the underlying xgboost model (for direct use with SHAP, etc.)
xgb_booster <- model_xgb$finalModel

# Variable importance (gain) from XGBoost
xgb.importance(model = xgb_booster)

# Get importance table
imp <- xgb.importance(model = xgb_booster)

# Take top 20 by Gain
imp_top <- imp[order(imp$Gain, decreasing = TRUE)[1:20], ]

# Horizontal bar chart
ggplot(imp_top, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "#2E86C1") +
  coord_flip() +
  labs(
    title = "Top 20 Features by Gain (XGBoost)",
    x     = NULL,
    y     = "Gain"
  ) +
  theme_minimal(base_size = 14)


# Scatter plot of Gain vs Cover
ggplot(imp, aes(x = Gain, y = Cover, size = Frequency)) +
  geom_point(alpha = 0.7, color = "#A569BD") +
  geom_text(aes(label = ifelse(rank(-Gain) <= 10, Feature, "")),
            hjust = 1.1, vjust = 0.5, size = 3) +
  scale_size_continuous(range = c(2,8)) +
  labs(
    title = "Feature Importance: Gain vs. Cover (size=Frequency)",
    x     = "Gain",
    y     = "Cover"
  ) +
  theme_minimal(base_size = 14)



# Compute cumulative gain
imp <- imp[order(imp$Gain, decreasing = TRUE), ]
imp$cumGain <- cumsum(imp$Gain) / sum(imp$Gain)

ggplot(imp, aes(x = 1:nrow(imp), y = cumGain)) +
  geom_line(color = "#E74C3C", size = 1) +
  geom_point(size = 2) +
  scale_x_continuous(breaks = seq(0, 20, by = 2)) +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Cumulative Gain Explained by Features",
    x     = "Number of Top Features",
    y     = "Cumulative Gain (%)"
  ) +
  theme_minimal(base_size = 14)


library(xgboost)
# Note: needs the importance matrix created by xgb.importance
# Base‐R XGBoost importance plot with a title
xgb.plot.importance(imp, top_n = 20, measure = "Gain", main = "XGBoost Feature Importance (Top 20 by Gain)")




# LightGBM

# Prepare data for LightGBM: LightGBM expects numeric matrix input
# We'll one-hot encode factors for LightGBM
train_mat <- model.matrix(Diagnosis ~ . - 1, data = train)  # one-hot encoding (no intercept)
train_label <- ifelse(train$Diagnosis == "AD", 1, 0)        # label as 0/1
test_mat  <- model.matrix(Diagnosis ~ . - 1, data = test)
test_label <- ifelse(test$Diagnosis == "AD", 1, 0)

# Create lgb.Dataset objects
dtrain <- lgb.Dataset(data = train_mat, label = train_label)

# Set LightGBM parameters for binary classification
params <- list(objective = "binary", metric = "auc", verbose = -1)

# Perform cross-validation to find optimal number of trees (nrounds) with early stopping
set.seed(123)
lgb_cv <- lgb.cv(params = params, data = dtrain, nfold = 5, nrounds = 1000, 
                 early_stopping_rounds = 10, verbose = 0)

best_iter <- lgb_cv$best_iter  # optimal number of boosting rounds from CV

# Train final LightGBM model with best number of rounds
model_lgb <- lgb.train(params = params, data = dtrain, nrounds = best_iter)

# Feature importance from LightGBM
lgb.importance(model = model_lgb)


# ====================================================================
# 4. Model Performance Evaluation
# ====================================================================

# Predict on test set with each model
pred_tree <- predict(model_tree, newdata = test)         # predicted classes
pred_rf   <- predict(model_rf, newdata = test)
pred_xgb  <- predict(model_xgb, newdata = test)

# For LightGBM, use the model to predict probabilities and then class
prob_lgb  <- predict(model_lgb, test_mat)                # probability of class 1 (AD)
pred_lgb  <- factor(ifelse(prob_lgb > 0.5, "AD", "NoAD"), levels = c("AD","NoAD"))

# Confusion matrices for each model
cm_tree <- confusionMatrix(pred_tree, test$Diagnosis, positive = "AD")
cm_rf   <- confusionMatrix(pred_rf,   test$Diagnosis, positive = "AD")
cm_xgb  <- confusionMatrix(pred_xgb,  test$Diagnosis, positive = "AD")
cm_lgb  <- confusionMatrix(pred_lgb,  test$Diagnosis, positive = "AD")

# Print accuracy and other stats
cm_tree$overall["Accuracy"]; cm_tree$byClass[c("Sensitivity","Specificity")]
cm_rf$overall["Accuracy"];   cm_rf$byClass[c("Sensitivity","Specificity")]
cm_xgb$overall["Accuracy"];  cm_xgb$byClass[c("Sensitivity","Specificity")]
cm_lgb$overall["Accuracy"];  cm_lgb$byClass[c("Sensitivity","Specificity")]


# Combine confusion matrix data for heatmap visualization
cm_list <- list(
  "Decision Tree" = as.data.frame(cm_tree$table),
  "Random Forest" = as.data.frame(cm_rf$table),
  "XGBoost"       = as.data.frame(cm_xgb$table),
  "LightGBM"      = as.data.frame(cm_lgb$table)
)

for(m in names(cm_list)) {
  cm_list[[m]]$Model <- m
  colnames(cm_list[[m]])[1:2] <- c("Prediction","Actual")
}
cm_all <- do.call(rbind, cm_list)


# Plot confusion matrix heatmaps for each model
ggplot(cm_all, aes(x = Actual, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5) +
  scale_fill_gradient(low = "grey", high = "violet") +
  facet_wrap(~ Model, ncol = 2) +
  scale_x_discrete(limits = c("AD","NoAD")) +          # Predicted: AD then NoAD
  scale_y_discrete(limits = c("NoAD","AD")) + 
  labs(title = "Confusion Matrices", x = "Actual Class", y = "Predicted Class") +
  theme_minimal(base_size = 14) +
  theme(
    plot.title    = element_text(face = "bold", hjust = 0.5),
    axis.title    = element_text(face = "bold"),
    axis.text     = element_text(size = 12)
  )

# Calculate Accuracy and AUC for each model on test set

# Probabilities for ROC (for tree and rf from caret models, we get probabilities)
prob_tree <- predict(model_tree, newdata = test, type = "prob")[, "AD"]
prob_rf   <- predict(model_rf,   newdata = test, type = "prob")[, "AD"]
prob_xgb  <- predict(model_xgb,  newdata = test, type = "prob")[, "AD"]
prob_lgb  <- prob_lgb  # already computed above

# Compute ROC AUC
roc_tree <- roc(test$Diagnosis, prob_tree, levels = c("NoAD","AD"), direction="<")
roc_rf   <- roc(test$Diagnosis, prob_rf,   levels = c("NoAD","AD"), direction="<")
roc_xgb  <- roc(test$Diagnosis, prob_xgb,  levels = c("NoAD","AD"), direction="<")
roc_lgb  <- roc(test$Diagnosis, prob_lgb,  levels = c("NoAD","AD"), direction="<")

auc_vals <- c(DecisionTree = auc(roc_tree), RandomForest = auc(roc_rf),
              XGBoost = auc(roc_xgb), LightGBM = auc(roc_lgb))
cat("AUC:\n"); print(round(auc_vals, 4))

# Summarize performance
results <- data.frame(
  Model = c("Decision Tree", "Random Forest", "XGBoost", "LightGBM"),
  Accuracy = c(cm_tree$overall["Accuracy"], cm_rf$overall["Accuracy"], 
               cm_xgb$overall["Accuracy"], cm_lgb$overall["Accuracy"]),
  AUC = c(as.numeric(auc_tree), as.numeric(auc_rf), as.numeric(auc_xgb), as.numeric(auc_lgb))
)
results

# Plot all ROC curves on one graph
roc_list <- list("Decision Tree" = roc_tree, "Random Forest" = roc_rf,
                 "XGBoost" = roc_xgb, "LightGBM" = roc_lgb)
ggroc(roc_list, aes = "colour", size = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  labs(title = "ROC Curves", x = "False Positive Rate", y = "True Positive Rate", color = "Model") +
  theme_minimal()

# ====================================================================
# 5. Model Interpretation and Visualization
# ====================================================================

## SHAP: Global Feature Importance

# Compute SHAP values for the XGBoost model on the training data
X_train_mat <- model.matrix(Diagnosis ~ . - 1, data = train)  # matrix of features
booster_feat_names<-xgb_booster$feature_names
X_train_mat <- X_train_mat[, booster_feat_names, drop = FALSE]
length(xgb_booster$feature_names)

shap_values <- shap.values(xgb_model = xgb_booster, X_train = X_train_mat)
# shap_values$shap_score holds the SHAP value matrix (rows = samples, cols = features)
# shap_values$mean_shap_score holds the mean absolute SHAP value for each feature

# Global feature importance (mean |SHAP value|)
shap_importance <- shap_values$mean_shap_score
shap_importance <- sort(shap_importance, decreasing = TRUE)
head(shap_importance, 10)  # show top 10 features by importance

# Prepare long-format data for SHAP plots
shap_long <- shap.prep(xgb_model = xgb_booster, X_train = X_train_mat)

# SHAP summary plot (beeswarm plot)

# 1. Generate the SHAP summary plot (no title argument)
plt.shap <- shap.plot.summary(shap_long)

# 2. Add your own title with ggtitle() (or labs())
plt.shap + 
  ggtitle("SHAP Summary: Feature Influence on XGBoost Model") +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold")
  )

