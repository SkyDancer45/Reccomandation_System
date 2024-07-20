
**Title:** Movie Recommendation System with Matrix Factorization in TensorFlow

**Description:**

This Jupyter Notebook implements a movie recommendation system using Matrix Factorization (MF) with TensorFlow. It utilizes the MovieLens dataset to learn latent factors representing user preferences and movie characteristics. These factors are then used to predict user ratings for unseen movies, providing recommendations.

**Data Citation:**

- F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. [https://doi.org/10.1145/2827872](https://doi.org/10.1145/2827872)

**Prerequisites:**

- Python 3
- Jupyter Notebook ([https://jupyter.org/](https://jupyter.org/))
- TensorFlow ([https://www.tensorflow.org/install](https://www.tensorflow.org/install))
- NumPy ([https://numpy.org/](https://numpy.org/))
- pandas ([https://pandas.pydata.org/](https://pandas.pydata.org/))
- tqdm ([https://tqdm.github.io/](https://tqdm.github.io/)) (optional, for progress bar)
- scikit-learn ([https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)) (optional, for KMeans clustering)

**Installation:**

```bash
pip install tensorflow numpy pandas tqdm scikit-learn  # Install if not already present
```

**Running the Notebook:**

1. Launch Jupyter Notebook in your terminal:

   ```bash
   jupyter notebook
   ```

2. Open this notebook (`Movie_Recommendation_MF.ipynb`) in your browser.

3. Run the code cells sequentially (top to bottom) by clicking the "Run" button (usually a play icon) or pressing `Shift+Enter`.

**Explanation:**

**1. Data Acquisition:**

   - The code downloads the MovieLens "ml-latest-small.zip" dataset using `curl`.
   - It extracts the data from the zip file and loads the movies and ratings dataframes using pandas.

**2. Data Exploration:**

   - The notebook prints the dimensions of the `movies_df` and `ratings_df` dataframes to understand the dataset size.
   - It retrieves and displays the first few rows of each dataframe to get a glimpse of the data.
   - Key metrics like the number of unique users and movies, sparsity of the rating matrix, and percentage of filled elements are calculated and printed.
   - These metrics highlight the challenge of working with sparse matrices in collaborative filtering tasks.

**3. Matrix Factorization Model:**

   - The `MatrixFactorization` class defines a PyTorch model with user and item embedding layers.
   - The model takes a user-movie pair as input and calculates the predicted rating using dot product between the corresponding embeddings.

**4. Dataset Preparation:**

   - The `Loader` class creates a PyTorch Dataset object for efficient data access during training.
   - It performs user and movie ID reindexing to create continuous IDs, easing model handling.
   - The ratings dataframe is transformed into a PyTorch tensor for training.

**5. Model Training:**

   - Hyperparameters like the number of epochs, learning rate, and number of latent factors are set.
   - The model is compiled with the MSE loss function and Adam optimizer.
   - A DataLoader wraps the `Loader` object for minibatch training.
   - The training loop iterates through the epochs, batches, and performs:
     - Forward pass: calculates predicted ratings for a batch of user-movie pairs.
     - Backward pass: computes gradients based on the MSE loss between predictions and true ratings.
     - Optimization step: updates model parameters using the optimizer.
   - The training loss is monitored and printed during each epoch.

**6. Model Analysis and Evaluation:**

   - After training, the latent factor matrices (user and item embeddings) are accessible for analysis.
   - The notebook (optional) could include code for exploring and visualizing the relationships between user and item embeddings.
   - KMeans clustering (optional) could be used on the item embeddings to group movies with similar characteristics. These clusters could provide insights into movie genres or themes.

**7. Movie Recommendation:**

   - To recommend movies for a user, the model predicts ratings for unseen movies using the user's embedding and the item embeddings of all movies.
   - Movies with the highest predicted ratings are considered the most relevant for the user.

