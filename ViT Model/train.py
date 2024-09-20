import torch
from torch import nn, optim
import os
from utils import save_experiment, save_checkpoint
from data import prepare_data
from vit import ViTForClassfication
import random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical 
import torchvision.transforms as transforms
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


def get_augmentations():
    return transforms.Compose([
        transforms.RandomRotation(10),  # Equivalent to rotation_range=20
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Equivalent to width_shift_range=0.2 and height_shift_range=0.2
        transforms.RandomAffine(degrees=0, shear=0.1),  # Equivalent to shear_range=0.2
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),# Equivalent to horizontal_flip=True
    ])

def augment_batch(batch):
    augmentations = get_augmentations()
    images, labels = batch
    augmented_images = []
    augmented_labels = []
    
    for img, lbl in zip(images, labels):
        
        augmented_img = augmentations(img)
        augmented_images.append(augmented_img)
        augmented_labels.append(lbl)
       

    augmented_images = torch.stack(augmented_images)
    augmented_labels = torch.tensor(augmented_labels)

    return augmented_images, augmented_labels



class Trainer:
    """
    The simple trainer.
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        

    def train(self, trainloader, testloader, epochs, config = None,steps_per_epoch = 0,augment = False,save_model_every_n_epochs=10):
       

        # Keep track of the losses and accuracies
        train_losses, test_losses, accuracies = [], [], []
        # Train the model
        for i in range(epochs):
            print("\nTraining epoch\n")
            train_loss = self.train_epoch(trainloader,steps_per_epoch,augment)
            print("\nEvaluating\n")
            accuracy, test_loss = self.evaluate(testloader)
            print("\nEvaluation Completed\n")
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
            if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0 and i+1 != epochs:
                print('\tSave checkpoint at epoch', i+1)
                save_checkpoint(self.exp_name, self.model, i+1)
        save_experiment(self.exp_name, config, self.model, train_losses, test_losses, accuracies)

            
    def train_epoch(self, trainloader, steps_per_epoch, augment):
        self.model.train()
        total_loss = 0
        trainloader_iter = itertools.cycle(trainloader)
    
    # Wrap the range with tqdm for the progress bar
        with tqdm(total=steps_per_epoch, desc='Training', unit='step') as pbar:
            for i in range(steps_per_epoch):
                batch = next(trainloader_iter)
                batch = [t.to(self.device) for t in batch]
                images, labels = batch
            
                if augment:
                    images, labels = augment_batch((images, labels))  
                images, labels = images.to(self.device), labels.to(self.device)
            
                self.optimizer.zero_grad()
            # Calculate the loss
                loss = self.loss_fn(self.model(images)[0], labels)
            # Backpropagate the loss
                loss.backward()
            # Update the model's parameters
                self.optimizer.step()
            
                total_loss += loss.item() * len(images)
            
            # Update the progress bar
                pbar.update(1)
    
        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in testloader:
                # Move the batch to the device
                batch = [t.to(self.device) for t in batch]
                images, labels = batch
                logits, _ = self.model(images)
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss
    
    @torch.no_grad()
    def predict(self, dataloader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                batch = [t.to(self.device) for t in batch]
                images, _ = batch  # Ignore labels, we are predicting
                logits, _ = self.model(images)
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())  # Collect predictions

        return np.array(predictions)

