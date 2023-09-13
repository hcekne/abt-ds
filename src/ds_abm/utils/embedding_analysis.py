# This module contains the necessary functions and classes to create embeddings

# Load and import libraries
import numpy as np
import pandas as pd
import random
#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



#----------------------------------------------------------------------------
## Data prep tasks and functions
#----------------------------------------------------------------------------



def enrich_with_false_observations(df1, customer_col, product_col):
    """
    Enrich a DataFrame with false observations for customer-product pairs.
    This is useful for generating negative samples when training an embeddings model.

    Parameters:
    ----------
    df1 : pd.DataFrame
        The input DataFrame containing the actual customer-product pairs.
    customer_col : str
        The column name in df1 representing the customer IDs.
    product_col : str
        The column name in df1 representing the product IDs.

    Returns:
    -------
    pd.DataFrame
        A new DataFrame containing both the original and the false observations,
        with an additional column 'true_shop' indicating the observation type (1 for true, 0 for false).

    """
    # Create an index of unique customer-product pairs
    customer_product_index = df1[[customer_col, product_col]].copy()

    # Extract unique customers and products
    unique_customers = list(customer_product_index[customer_col].drop_duplicates())
    unique_products = list(customer_product_index[product_col].drop_duplicates())

    # Generate random customer and product pairs
    random_customers = random.choices(unique_customers, k=len(df1) * 2)
    random_products = random.choices(unique_products, k=len(df1) * 2)

    # Create a DataFrame from the random pairs
    random_customer_products = pd.DataFrame({
        'rand_cust': random_customers,
        'rand_prod': random_products
    })

    # Remove duplicate random pairs
    random_customer_products = random_customer_products.drop_duplicates().reset_index(drop=True)

    # Identify random pairs that actually exist in the original data
    both = random_customer_products.merge(
        customer_product_index.drop_duplicates().reset_index(drop=True),
        left_on=['rand_cust', 'rand_prod'],
        right_on=[customer_col, product_col]
    )

    # Filter out the random pairs that are in the original data
    random_customer_products = random_customer_products.merge(both, how='left')
    random_customer_products = random_customer_products[random_customer_products[customer_col].isnull() == True]
    random_customer_products.drop(columns=[customer_col, product_col], axis=1, inplace=True)
    random_customer_products.reset_index(drop=True, inplace=True)
    random_customer_products = random_customer_products.sample(len(df1))
    random_customer_products.columns = [customer_col, product_col]

    # Add a 'true_shop' column to indicate true and false observations
    random_customer_products['true_shop'] = 0
    customer_product_index['true_shop'] = 1

    # Combine true and false observations
    df2 = pd.concat([random_customer_products, customer_product_index])
    df2.reset_index(drop=True, inplace=True)

    # Shuffle the DataFrame
    df2['random_sorting_numbers'] = np.random.uniform(size=len(df2))
    df2.sort_values(['random_sorting_numbers'], inplace=True)
    df2.reset_index(drop=True, inplace=True)
    df2.drop(columns=['random_sorting_numbers'], inplace=True)

    return df2


#----------------------------------------------------------------------------
# ------- Embedding model definitions----------------------------------------
#----------------------------------------------------------------------------

class PreTrainEmbedding(nn.Module):
    def __init__(self, n_customers, n_products, n_factors):
        super(PreTrainEmbedding, self).__init__()
        self.cust_embedding = nn.Embedding(n_customers, n_factors)
        self.prod_embedding = nn.Embedding(n_products, n_factors)
        self.out = nn.Linear(1, 1)  # Since the dot product results in a single value, the output layer has 1 input feature
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        custs = self.cust_embedding(x[:,0])
        prods = self.prod_embedding(x[:,1])
        # Element-wise multiplication and sum to get the dot product
        dot_product = torch.sum(custs * prods, dim=1, keepdim=True)
        x = self.out(dot_product)
        x = self.sigmoid(x)
        return x



# Define the second model architecture
class SecondEmbeddingModel(nn.Module):
    def __init__(self, n_customers, n_products, n_factors):
        super(SecondEmbeddingModel, self).__init__()
        self.cust_embedding = nn.Embedding(n_customers, n_factors)
        self.prod_embedding = nn.Embedding(n_products, n_factors)
        self.out = nn.Linear(1, 1)  # Since the dot product results in a single value, the output layer has 1 input feature
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        custs = self.cust_embedding(x[:,0])
        prods = self.prod_embedding(x[:,1])
        # Element-wise multiplication and sum to get the dot product
        dot_product = torch.sum(custs * prods, dim=1, keepdim=True)
        x = self.out(dot_product)
        return x

def model_training_loop(embed_model, train_loader, criterion, optimizer, epochs=20):
    for epoch in range(epochs):
        # batch_losses = []
        for i, (cust, prod, y) in enumerate(train_loader):
            y_pred = embed_model(torch.stack([cust, prod], dim=1).long())
            loss = criterion(y_pred.squeeze(), y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
#----------------------------------------------------------------------------
#---Functions for using the models and creating the embedddings
#----------------------------------------------------------------------------





def create_double_embedding(df1,n_factors=20, pre_train_model_epochs=20, second_model_epochs=20, 
                            customer_col='CustomerIndex', product_col='ProductIndex', quantity_col='Quantity'):
    """
    Creates embeddings of customers and products. Uses a double embedding architecture and 
    first creates the embeddings based on a boolean value of whether a customer and product
    combination has been observed within the data. Then loads those embeddings into another
    model that uses the quantity of product purchased to be create the embebbing model.

    Parameters:
    ----------
    df1 : pd.DataFrame
        The input DataFrame containing the actual customer-product pairs.
    n_factors : int
        The number of embedding dimensions in the embedding models
    pre_train_model_epochs : int
        The number of epochs to train the first model for
    second_model_epochs : int
        The number of epochs to train the second model for
    customer_col : str
        The column name in df1 representing the customer index.
    product_col : str
        The column name in df1 representing the product index.
    quantity_col : str
        The column used to determine product quantity for the second model.

    Returns:
    -------
    customer_embeddings_2 : np.array
        A numpy array with embeddings for the customers
    product_embeddings_2 : np.array
        A numpy array with the product embeddings

    """

    
    #----------# 0. Perform needed data analysis #----------------#
    n_customers = df1[customer_col].nunique()
    n_products = df1[product_col].nunique()

    df_pre_train = enrich_with_false_observations(df1, customer_col, product_col)

    # Extracting relevant columns for the first pretraining embedding model
    customer_indices_1 = torch.tensor(df_pre_train[customer_col].values, dtype=torch.long)
    product_indices_1 = torch.tensor(df_pre_train[product_col].values, dtype=torch.long)
    true_shop = torch.tensor(df_pre_train['true_shop'].values, dtype=torch.float32)
    
    # extracting the relevant columns for the second embedding model
    customer_indices_2 = torch.tensor(df1[customer_col].values, dtype=torch.long)
    product_indices_2 = torch.tensor(df1[product_col].values, dtype=torch.long)
    quantities = torch.tensor(df1[quantity_col].values, dtype=torch.float32)


    #----------# 1. Model #----------------#
    
    # Initialize the pre-training model
    pre_training_model = PreTrainEmbedding(n_customers, n_products, n_factors)

    # Data Loader (adjust batch_size as needed)
    train_data = TensorDataset(customer_indices_1, product_indices_1, true_shop)
    train_loader_1 = DataLoader(train_data, batch_size=1024, shuffle=True)
    # Training Looop
    model_training_loop(embed_model=pre_training_model, train_loader=train_loader_1,
                        criterion=nn.BCELoss(), # Binary Cross-Entropy Loss, optimizer:criterion_pre_train,
                        optimizer=optim.Adam(pre_training_model.parameters(), lr=0.01),
                        epochs=pre_train_model_epochs)

    #----------# 2. Model #----------------#
            
    # Initialize the model
    second_model = SecondEmbeddingModel(n_customers, n_products, n_factors)
    # Load state from first model
    second_model.load_state_dict(pre_training_model.state_dict(), strict=False)    

    # Data Loader (adjust batch_size as needed)
    train_data = TensorDataset(customer_indices_2, product_indices_2, quantities)
    train_loader_2 = DataLoader(train_data, batch_size=1024, shuffle=True)
    # Training loop (adjust epochs as needed)
    model_training_loop(embed_model=second_model, train_loader=train_loader_2, 
                        criterion=nn.MSELoss(), 
                        optimizer=optim.Adam(second_model.parameters(), lr=0.01),
                        epochs=second_model_epochs)

    #----------# 3. Generating output #----------------#
                                
    # Extract embeddings from the 2 model
    customer_embeddings_2 = second_model.cust_embedding.weight.detach().numpy()
    product_embeddings_2 = second_model.prod_embedding.weight.detach().numpy()


    return customer_embeddings_2, product_embeddings_2
 