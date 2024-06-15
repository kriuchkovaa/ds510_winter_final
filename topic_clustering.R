# Step 0: For loading required libraries

# Update to the latest version of R if needed
# install.packages("installr")
# require(installr)
# updateR()

# install.packages("readtext")    # for extracting text
#("tm")           # for text pre-processing
#install.packages("textstem")     # for lemmatization


library(readtext)
library(tidyverse)
library(tm)
library(textstem)
library(wordcloud)
library(cluster)

# Step 1: Export all textfiles into a dataframe
txt_files <- list.files(".", pattern = "\\.txt$", full.names = TRUE)
df <- readtext(txt_files, encoding = "UTF-8")
print(str(df))

# Assuming 'df' is your dataframe
write.csv(df, file = "my_dataframe.csv")


# Remove .txt extension for each document ID to produce a topic
df <- df %>%
  mutate(doc_id = sub("\\.txt$", "", doc_id))

# Show results
print(df, n = 21)
print(nrow(df))
print(attributes(df))

# # Step 2: Text Pre-processing

# Create a corpus for managing our text data
corpus <- VCorpus(DataframeSource(df))

# Create a custom function that encompases all text transformations
# Pay close attention to sequence of operations!

text_preprocessing <- function(x) {
  x <- tolower(x) # Convert to lowercase
  x <- stripWhitespace(x) # Strip whitespaces
  x <- removeNumbers(x)   # Remove numbers
  x <- gsub('http\\S+\\s*', ' ', x) # Remove URLs
  x <- gsub('#\\S+', ' ', x) # Remove hashtags
  x <- gsub('[[:cntrl:]]', ' ', x) # Remove controls and special characters
  x <- gsub("[[:punct:]]", " ", x) # Replace punctuation with spaces
  return(x)
}

# Apply changes
corpus <- tm_map(corpus, content_transformer(text_preprocessing))

# Define custom stopwords
custom_stopwords <- c("subject", "faq", "resources", "modified", "archive", "version", "newsgroup", "document", "id", "ffrf", "mailrc", "usr", "kfu", "jah")

all_stopwords <- c(stopwords("english"), custom_stopwords)
corpus <- tm_map(corpus, removeWords, all_stopwords)

# Lemmatization
corpus <- tm_map(corpus, content_transformer(lemmatize_strings))
print(substr(corpus, 1, 500))

# Create a weighted TF-IDF
dtm <- DocumentTermMatrix(corpus)
dtm_tfidf <- weightTfIdf(dtm)

print(dtm_tfidf)

# Step 3: Topic Modeling using K-Clustering
k_model <- kmeans(dtm_tfidf, centers = 7)

# Find most frequently used words for a given cluster
get_top_words <- function(dtm, centers, cluster_of_interest, n = 10) {
  cluster_center <- centers[cluster_of_interest,]
  all_centers_mean <- colMeans(centers[-cluster_of_interest,])
  word_scores <- cluster_center - all_centers_mean
  top_indices <- order(word_scores, decreasing = TRUE)[1:n]
  
  terms <- colnames(dtm)
  top_terms <- terms[top_indices]
  
  word_freqs <- word_scores[top_indices]
  names(word_freqs) <- top_terms
  
  return(word_freqs)
}

# Test the function for a selected cluster 
print(get_top_words(dtm_tfidf, k_model$centers, 5))


# Using an elbow method for determining the most optimal value of k

library(cluster)

dtm_subset <- removeSparseTerms(dtm_tfidf, 0.95)   # more topic specific terms  
tm::inspect(dtm_subset)


# Define the max number of clusters to test
max_clusters <- 19

# wcss: within cluster sum of squares - qualifies the variance of data withing a cluster
# Initialize a vector to store the WCSS values
wcss <- numeric(max_clusters)

nrow(dtm_subset)

# Loop over several values of k (number of clusters)
for (k in 1:max_clusters) {
  set.seed(10) #set seed for reproducibility
  kmeans_model <- kmeans(dtm_subset, centers=k, nstart=10) #Fit the model for iteration
  wcss[k] <- kmeans_model$tot.withinss  
}

# Plot the results
plt_elbow  <- ggplot(data.frame(k = 1:max_clusters, wcss = wcss), aes(x = k, y = wcss)) +
  geom_line() +
  scale_x_continuous(breaks = c(3,5,7,9,11,13,20))+
  geom_point() +
  ggtitle("Elbow Method") +
  xlab("Number of clusters")
ylab("WCSS")

windows(); plt_elbow

# Re-run k-means clustering on the subset DTM
# Try k = 5 through k = 9
k_model_7 <- kmeans(as.matrix(dtm_subset), centers = 7)
top_words <- get_top_words(dtm_subset, k_model_7$centers, 1)

k_model_6 <- kmeans(as.matrix(dtm_subset), centers = 6)
top_words_k6 <- get_top_words(dtm_subset, k_model_6$centers, 1)

k_model_8 <- kmeans(as.matrix(dtm_subset), centers = 8)
top_words_k8 <- get_top_words(dtm_subset, k_model_8$centers, 1)

k_model_5 <- kmeans(as.matrix(dtm_subset), centers = 5)
top_words_k8 <- get_top_words(dtm_subset, k_model_5$centers, 1)

k_model_9 <- kmeans(as.matrix(dtm_subset), centers = 9)
top_words_k8 <- get_top_words(dtm_subset, k_model_9$centers, 1)

# Now lets create custom word clouds to visualize our findings
library(RColorBrewer)

# Define a color palette
color_palette <- colorRampPalette(brewer.pal(8, "Paired"))(200)

# Shuffle colors to introduce more variety 
# Wordcloud tends to create random clouds with homogenous color choices
shuffle_colors <- function(colors) {
  sample(colors, length(colors))
}

# Word cloud generator
generate_word_clouds <- function(k_model, dtm_subset, file_name) {
  num_clusters <- nrow(k_model$centers)
  # For determining layout dimensions
  layout_matrix <- ceiling(sqrt(num_clusters))

  # Allow to save as an image
  png(file_name, width = 600, height = 600)

  # Set up partitioned plot
  par(mfrow = c(layout_matrix, layout_matrix), mar = c(2, 2, 2, 2))
  
  for (cluster_number in 1:num_clusters) {
    word_freqs <- get_top_words(dtm_subset, k_model$centers, cluster_number, 10)
    colors_selector <- shuffle_colors(color_palette)
    
    wordcloud(
      words = names(word_freqs),
      freq = word_freqs,
      max.words = 100,
      random.order = FALSE,
      scale = c(3, 0.5),
      colors = colors_selector,
      main = paste("Cluster", cluster_number)
    )
  }
  
  # Reset to default & finalize the saving
  par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)
  dev.off()
}

# Generate word cloud partitions
generate_word_clouds(k_model_7, dtm_subset, "Word clouds for k7.png")

generate_word_clouds(k_model_6, dtm_subset, "Word clouds for k6.png")

generate_word_clouds(k_model_8, dtm_subset, "Word clouds for k8.png")

generate_word_clouds(k_model_5, dtm_subset, "Word clouds for k5.png")

generate_word_clouds(k_model_9, dtm_subset, "Word clouds for k9.png")

# Silhouette scores

# Calculate silhuette score
silhouette_score <- function(model, data_matrix) {
  dist_matrix <- dist(data_matrix)
  ss <- silhouette(model$cluster, dist_matrix)
  mean(ss[, 3])
}

# Create a list of existing models for iteration
models <- list(k_model_5, k_model_6, k_model_7, k_model_8, k_model_9)
data_matrix <- as.matrix(dtm_subset)

# Compute silhouette score
sil_scores <- sapply(models, silhouette_score, data_matrix = data_matrix)

# Identify the centers
centers <- c(5, 6, 7, 8, 9)

# Plot the results
windows(); plot(centers, sil_scores, type = 'b', xlab = 'Number of clusters', ylab = 'Average Silhouette Scores', frame = FALSE, col = "blue", pch = 19)
title("Silhoette Scores: k values 5-9")

