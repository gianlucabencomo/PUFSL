This text file will contain journal entries for the project I am pursuing in my Probabalistic Modeling course at 
Princeton University.  The format will be rather informal, and I will be making weekly entries (at the very least).

02.07.2022
The abstract is due today.  I wrote an initial version last night, but I feel that improvements can be made. In
the past few days, I have read several papers on few-shot/one-shot learning (FSL/OSL). I am certain that I would 
like to do a project related to drug discovery, since research there is scarce for FSL/OSL.  Many exciting FSL/OSL 
methods have emerged in the past few years but they have not really be applied to drug discovery.  This creates 
novelty in my project: adapting a state-of-the-art OSL/FSL method to perform drug discovery.  I need to 1) settle
on a method to adapt, and 2) settle on datasets to work with.  Hopefully, I can continue to read papers and have
these two items set in stone.  My abstract, as it stands, contains a general idea of methods/datasets but I am
uncertain as to whether or not I will stick to my currently written direction.

03.11.2022
It has been some time since my last journal entry and commit.  Over the past week, I have done a substantial 
amount of work.  I familiarized myself with the deepchem library, I wrote various wrapper functions to allow me
to more easily access the deepchem API and build/train models, I ran some initial tests and trained some initial 
models, and I read a ton of papers on MAML, drug discovery, and meta-learning in general.  After spending a lot of
time with the deepchem library, I have concluded that it might not be the best choice for myself.  I am finding 
the documentation confusing, and deepchem, as a whole, seems to be a blackbox.  Additionally, when using the API,
so many expection messages pop up, which leads me to believe that there many be some data mishandling that would 
take too long to investigate.  I think MoleculeNet is fantastic, and I plan to take advantage of their data, but
avoiding the deepchem library as a whole seems like the right move for this project.  
I also spent a considerable amount of time thinking about how I can apply MAML to this project, but I recently
turned away from that idea.  Prototypical networks are more simple, and more appropriate for this project. I plan
on doing one-shot/few-shot drug discovery with prototypical networks, and then moving on to MAML/ProtoMAML if time
permits. 
The plan as of right now is to work directly with the PCBA, Tox21, ToxCast, and MUV CSV files kindly provided by 
MolNet and code a featurizer to convert the SMILES strings into graph ConvNets compatiable with Pytorch.  Then,
I can train a prototypical network!
 
