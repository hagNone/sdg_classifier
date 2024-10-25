SDG_goals classifier based on the company overview.

Here we collected the data from different sources like company websites and company pitch videos.
The data is used to train the modelm, before training there is thing where we need to clean/preprocess the rew data so it would be fit well to the model.
We here removed all the stop words punctuations etc.
We then used OnevsRest classifier cause there are multiclass labels in the data where using this would be easier to classify and the accuracy of the models have been analysed with ploting the graphs
For better results we even used the pretrained model from hugging face called DistilBert which is the lighter version of Bert.
We even done ploting with meta data analysis and ploting the realtionship between sdg and companies like how would it effect it 
