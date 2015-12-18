# Online Hate Speech

This is a project aimed at analyzing online hate speech to develop a predictive model which could flag whether a user's comments may be hate speech, and against what group (against specific religions, skin colors, gender, sexual orientation and  size).

Hate speech is broadly defined as speech advocating incitement to harm based on the target's membership in a group. Social media, news, and technology companies have struggled to find the balance in their online spaces, between allowing wide discussion, and preventing their users from becoming alienated and threatened as a result of a few other users' hate speech. The model developed here could help streamline the online comment moderation process.

In production, the model could flag potentially inflammatory comments and/or users who may be becoming hateful, and prevent the comment from being posting until the comment(s) are reviewed by human moderaters.

Alternatively, this could provide a public label. A user's avatar color could change, or have letters superimposed on it, in response to the model predicting hateful speech. A user could have an indication of hating women, for example, and other users of the site could easily identify this user as hating women, and choose to weight the hateful user's comments accordingly. This would require less human moderation, but may have unintended consequences, such as hateful users find each other via their hateful labels.

### Data Source
The primary data source for this project is the May 2015 reddit corpus available from [kaggle.com](https://www.kaggle.com/c/reddit-comments-may-2015/data).
