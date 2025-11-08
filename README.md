# Overview

Welcome to the First Challenge of the Artificial Neural Networks and Deep Learning course for A.Y. 2025-2026! This is your chance to put what you’ve learned to the test and compete with your classmates!

- **Start:** 2 days ago
- **Close:** 9 days to go

# Description

In this assignment, you will receive a collection of multivariate time series. Each sequence consists of 180 time steps and contains measurements from several input channels. The objective of this task is to perform time series classification — that is, to assign each entire sequence to one of three possible classes.

⚠️ Please note that, as specified in the Rules section, the use of pretrained models is forbidden. All neural networks must be trained from scratch, without relying on any pre-existing architectures or pretrained weights.

# How To Participate

## Registration

To take part in this homework, follow these steps:

1. Register to this Google form. Please note that your Kaggle account must be linked to your institutional email (name.surname@mail.polimi.it) and your Kaggle nickname must match the one provided in the form.
2. Apply to this competition.
3. Start designing your solution.
4. Submit your solution as a `.csv` file.

## Teams

Once you join the competition, it is mandatory to create or add your team in the `Team` tab.

Please note that groups have a minimum of 3 and a maximum of 4 members.

Students looking for additional team members should wait 24-48h after the form deadline so that we can finalise group assignments. We will notify those students by email and publish the official group assignment document in the shared folder of the exercise sessions.

# Evaluation

For each team, the evaluation of this challenge comprises the final leaderboard score and the report score. The combined total of these scores will range from 0 to 5 and will be added to your written exam score. We will give 0.5 additional points for exceptional works.

## Leaderboard Metrics

Your models will be evaluated according to the F1-Score, computed as given the set of classes, the ground truth, and the model predictions.

To prevent overfitting, during the entire competition period your submissions will be evaluated on a specific subset of the test set. The final results evaluated on a separate test subset will be available on November 17, 23:59.

Please remember that for this Homework, pretrained models are forbidden. All neural networks must be trained from scratch, without relying on existing trained architectures.

## Submission File

You must submit a `.csv` file containing the predictions. For each ID in the test set, you must predict the class. The file should contain a header and have the following format:

```
sample_index,label
000,high_pain
001,high_pain
002,high_pain
```

etc.

## Final Report

Before the end of the competition (November 17, 23:59), you are required to submit a report detailing your activities. The report should include:

- Information about team members.
- The development process of your model.
- The ideas behind your final solution.
- All necessary information to demonstrate your work.

The final report is part of your homework grade, alongside your results from the Leaderboard.

The report must be a PDF file with a maximum length of 3 pages excluding references. It is mandatory to use the template provided here: Latex Template on Overleaf (Read Only). This is to ensure a cohesive structure for all your reports. Please note that any change to the template and any additional page after the 3 allowed will result in penalties.

To submit your report, one member of the team must send an email to `airlab.official.polimi@gmail.com`. Attach the report to this email.

In addition to the report, you must attach to the email a zip file containing one (or more) notebook(s) that precisely demonstrate the models described in the report. This includes the code for creating and training the models, along with the local performance results, visible in the notebook output cells. The notebooks must be fully executed, with all outputs visible. Ensure that your code is well-commented and reproducible, highlighting the roles of functions and variables, as well as the main steps.

⚠️ The deadline for sending the email is November 17, 23:59.

The email must adhere to the following content. No additional text is needed. Emails that do not follow this format will not be considered:

```
Subject: AN2DL25 - Challenge 1

Body:

GROUP_NAME: [Your group name]

STUDENT_1_NAME: [Name Surname]
STUDENT_1_ID: [matricola]
STUDENT_1_EMAIL: [email]

STUDENT_2_NAME: [Name Surname]
STUDENT_2_ID: [matricola]
STUDENT_2_EMAIL: [email]

STUDENT_3_NAME: [Name Surname]
STUDENT_3_ID: [matricola]
STUDENT_3_EMAIL: [email]

STUDENT_4_NAME: [Name Surname] (if applicable)
STUDENT_4_ID: [matricola] (if applicable)
STUDENT_4_EMAIL: [email] (if applicable)

REPORT_FILENAME: [Your report filename .pdf]
ZIP_FILENAME: [Your zip filename .zip]
```

# Miscellanea Information

## Using the Competition Forum

We ask that you avoid sending direct emails to the course team or professors for code-specific issues. Instead, please post any questions or clarifications in the Discussion tab, so that they are available to everyone. The forum is a space for collaboration, as this challenge is about learning how to train deep learning models, not competing against each other!

Feel free to share ideas, ask questions, and support one another.

## The Logbook is Back

If you would like to receive suggestions, strategy proposals and generally take part in the long-running Lomurno's Turkey Game 🦃, you can keep an eye on this document.

## GPU Usage Recommendations

To tackle this challenge, use the frameworks we discussed during practical sessions (TensorFlow, Keras).

Google Colaboratory offers free cloud GPU access with usage limits, so review their guidelines for more details. Kaggle also provides access to free GPUs, albeit with some limitations. For local GPU users, ensure that your device has the correct GPU drivers installed for your CUDA version.

Good luck, and may the best model win!

