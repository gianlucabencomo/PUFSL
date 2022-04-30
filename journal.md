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

02.14.2022
Still reading a ton of papers and familiarizing myself with the course/material.  I have been playing around with
some stuff like MAML and other techniques.

02.17.2022
I have been playing around with some example notebooks that are offered by the deepchem library.  I think the 
deepchem people have done some cool work but the library itself seems to be poorly suited for ML folks. The whole
thing is one big black box! There are some helpful features but I am undecided as to whether or not deepchem can
be of use.  I think deepchem would be great if I was doing conventional out-of-the-box methods, but I am not. I
will continue exploring.

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

03.12.2022
I am getting a much better idea of how the implementation will go. I am almost done with featurizing my 
smile strings as such that they are represented by graphs and compatitble with pytorch.  I will be using 
the pytorch geometric library to do most of the modeling and heavy lifting.

03.12.2022
I have pretty much completed all the code neccessary to take SMILES strings and their respective labels and 
convert that into graph data with associated labels.  The form is compatiable with pytorch.  There is one small
snag, however.  I keep getting a "zsh: segmentation fault  python" error that is preventing me from testing my
code and making sure that it is good to go.  Regardless, I have been substaintial progress and think I am at a 
good stopping point for now.  After I make sure that all the data handling works, it will be very easy to train
a graph conv net with GAT/GCN layers and try to get a strong model for each task individually.  I can get into
prototypically networks shortly after.

03.21.2022
I have begun writing my milestone report and the repository as it is today represents the progress that I will be
reporting on.  Today, I added code that replicates the results of prototypical networks as reported in their paper.
It is important to note that this code is not my own, but I will be using it as a guide when writing my own 
version.  I plan on adapting the code such that prototypical networks can predict new chemical tasks rather than
new classes.

04.07.2022
I have done a pretty sizeable amount of work in the past two weeks, so this entry will be long.  Prior to
submitting my milestone report, I replicated the results originally shown in the prototypical networks paper.
That code will temporarily be in my github repo until the project is complete, at which points I am going to do
some spring cleaning.  
More recently, I have been working towards adapting protonets to work with graphs and the datasets at hand. There
are a ton of considerations that I will need to make in the come days.  First, datasets.  There are a ton of 
missing entries in my 4 datasets, and pcba is much too large.  I need to consider how I should be delegating my
training/test tasks (random split or predetermined) and I need to find a way to appropriately handle all of the
missing entries for specific tasks.  Right now, I am thinking about having the first 80% of tasks always be the
training set and the last 20% of tasks be the test set (no validation set).  Other papers have done it this way
as well.  When I am training, I will randomly select a task for each episode and randomly select a compound. If
the compound doesn't have any data for that task, I will sample again.  The query set will be the same deal and
I will make sure that there is no overlap between support and query.  Right now, my implementation as it is 
agrees with what I have just wrote.  I also need to consider the ratio between 0/1 results in each task.  If
95% of tox21 is 0 then anything less than a 95% prediction acc is terrible.  I need to do a little bit of 
investigation there.  The second consideration is understanding graph attention networks and properly 
implementing them.  I need to do a deep paper dive and really get a good understanding before I finalize anything
but I currently think I will be using two GATConv layers.  Once I have this architecture set up and figure out
how to embed all compounds identically then I will be all set for implementing the rest of the protonet.  All
I really have to do is fix my data concerns but, truthfully, I could also work with other graph data that is 
just as interesting if it turns out that this data is not very good.  
Since running and debugging these training episodes will take some time, I should try to finish all of this
as soon as possible.  Hopefully, I can put in another solid 10 hours this weekend and then finish up the rest
of the implmentation in the week that follows. 

04.16.2022
Right now, featurizer.py is complete, data.py runs but there may be some bugs, config.py is well-organized, and
main.py is coming along.  In data.py, I am successfully able to split the tasks into training and testing tasks.
However, I am a little bit confused on how the nature of my embeddings should go and how I should run the train
loop.  This is going to take me a pretty sizeable amount of time to figure out, and I am anticipating a ton of 
bugs.  

04.19.2022
I finished about 99% of the implementation!  I am now able to train/test few-shot models on any chemical/graphical
dataset.  The work is not completely done, however.  I still need to 1) write the final paper, 2) prepare a final
presentation, 3) create nice readme for the repository, 4) fix the loss function so I can accept support sets with
different sizes of pos/neg, 5) consider having a random train/val/test split instead of a fixed one, 6) do a t-sne
visualization of the embeddings as well as think about and implement other types of visualization, 7) optimize the 
embedding function because, while it does work and leads to okay performance, it does not match the state of the art
yet.  If I can optimize the embedding function as such that I signficantly improve accuracy, I could be well on my
way to a publication, 8) do a simple random forests few-shot implementation so I have something to compare to, 9)
any other improvements/modifcations to the code!
Thankfully, I think I can start writing the final paper and complete 80% of it without having to code anything else.
The parts of the paper I cannot yet complete are 1) model comparison (to random forests), 2) describing and detailing
the architecture of the embedding function, because I have not finalized it yet, 3) results in general.  I can most
certainly write the abstract in full, the introduction in full, the conclusion in full, methods up until the point
where I describe the embedding function, and the results have to be empty for now.
For figures and tables, I think I want: 1) diagram of prototypical networks from snell, 2) t-sne for results, 3) table
with accuracies within a confidence interval for 10-shot, 5-shot, 3-shot, and 1-shot.

04.23.2022
The architecture of the embedding function is something that needs to be optimized, but it can be saved for further
exploration.  I will mention it in the conclusion.  I also need to fix the loss function so I can have support sets
with different pos/neg.  That can also be saved for later.
Right now, I need to 1) add the SIDER dataset so I can perform infences with tox21, muv, and SIDER for the report.
2) do a simple random forests implemention for few-shot learning. 3) implement t-sne so we can visualize the
embeddings.
After that is done, I can write the report.

04.25.2022
I have decided that I will not be implementing any of the random forests or t-sne ideas for the purpose of this
final report.  I have put in an insane amount of hours into this project, and believe I went well beyond the lower
bound of work needed to get a 100% on the final report.  I successfully implemented several very tricky ideas and did
a ton of testing with them.  I think the work that I did here is both interesting and meaningful, and my time would be 
better spent studying new material / working on finals.  
The report is almost done and I think it is shaping up quite nicely.  I did a ton of testing with two datasets (tox21,
MUV) and acheived really good results.  While the implementation I currently have can be improved, I still think it is
very clean and very good.  If I turn this project into a paper, there will be some more work that I will have to do that
is going to be detailed in my final project report.  I will probably make two more entries after this one, and maybe clean
up my code a little bit more, but for now I feel really good about the work I have done.

04.26.2022
Just made some very minor code adjustments and I think everything is ready for submission.  I am going to commit the code
and leave ./src static until my project has been graded.  At some point over summer, I will revisit the repo and try to
turn the project into something that is publishable, with a few minor adjustments and additions.
The only other thing that I have to do after this next commit is finalize my final report and presentation slides.  I
will add the document and slides to ./doc and I believe the project will be in perfect shape for submission there after.
In addition, I will likely make 1-2 more journal entries.

04.30.2022
I have been hard at work and believe this will be my final entry for this project and class. I just finalized my final paper
and placed the tex files and pdf under ./doc.  I am almost done compiling my final presentation and I feel good about the
work that I was able to do.  Of course, I still need to write a nice README and do some general housekeeping for the project
that is completely outside of the scope of this course.  Additionally, I would like to eventually bring the project to a 
publishable state, as I have said before.  Nonetheless, this will be my final entry because I am more of a handwritten 
journal person and I would like to transition back to my physical research journal.
...
To Dr. Adji Bousso Dieng and the 513 TAs, thank you for a great semester, insightful lectures, and guidance on assignments
and projects.
