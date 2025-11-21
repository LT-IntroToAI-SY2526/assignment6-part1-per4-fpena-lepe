# Assignment 6 Part 1 - Writeup

**Name:** ___Franco Pena-Lepe
**Date:** __11/21/2025
---

## Part 1: Understanding Your Model

### Question 1: R² Score Interpretation
What does the R² score tell you about your model? What does it mean if R² is close to 1? What if it's close to 0?

**YOUR ANSWER:**
The R² score tells you how much of the variation in your  variable  is explained by your other variable which in this case is hours studied.



---

### Question 2: Mean Squared Error (MSE)
What does the MSE (Mean Squared Error) mean in plain English? Why do you think we square the errors instead of just taking the average of the errors?

**YOUR ANSWER:**
MSE tells you how far off the model’s predictions are from the actual values, with larger mistakes counted more heavily. We square the errors because it punishes the larger mistakes more than the smaller ones. 



---

### Question 3: Model Reliability
Would you trust this model to predict a score for a student who studied 10 hours? Why or why not? Consider:
- What's the maximum hours in your dataset?
- What happens when you make predictions outside the range of your training data?

**YOUR ANSWER:**
I would trust it only somewhat but not fully.The maximum study time in the dataset is about 9.8 hours, and most of the data is in the 7–8 hour range. So predicting a score for someone who studied for 10 hours would mean the prediction might be off because there is no data for students who studied 10 hours. When you predict outside your training data your predictions might be off.



---

## Part 2: Data Analysis

### Question 4: Relationship Description
Looking at your scatter plot, describe the relationship between hours studied and test scores. Is it:
- Strong or weak?
- Linear or non-linear?
- Positive or negative?

**YOUR ANSWER:**
From the scatter plot, the relationship between hours studied seems to be strong, linear, and positive. There is some variation around 8-9 hours but overall its strong linear and positive. 



---

### Question 5: Real-World Limitations
What are some real-world factors that could affect test scores that this model doesn't account for? List at least 3 factors.

**YOUR ANSWER:**
Amount of sleep before the exam

Student motivation or stress level

Quality of studying or teaching (study materials, tutoring, environment)


---

## Part 3: Code Reflection

### Question 6: Train/Test Split
Why do we split our data into training and testing sets? What would happen if we trained and tested on the same data?

**YOUR ANSWER:**
We split data into training and testing sets so we can see how well the model performs on new data.
If we trained and tested on the same data, the model could seem “perfect” because it memorized the training data. It would perform poorly on new students or real-world cases.


---

### Question 7: Most Challenging Part
What was the most challenging part of this assignment for you? How did you overcome it (or what help do you still need)?

**YOUR ANSWER:**
The most challenging part was understanding the tools and environment like anaconda and panda and how the code worked. I overcame this by reading through the documentation, experimenting with the code, and reviewing what each library function does.



---

## Part 4: Extending Your Learning

### Question 8: Future Applications
Describe one real-world problem you could solve with linear regression. What would be your:
- **Feature (X):** 
- **Target (Y):** 
- **Why this relationship might be linear:**

**YOUR ANSWER:**
 Predicting how long it takes someone to commute based on the distance they travel.

Feature (X): Distance from home to work (in miles or km)
Target (Y): Commute time (in minutes)


In many situations, the farther someone lives from work, the longer the commute. So this relationship would likely be linear.



---

## Grading Checklist (for your reference)

Before submitting, make sure you have:
- [ ] Completed all functions in `a6_part1.py`
- [ ] Generated and saved `scatter_plot.png`
- [ ] Generated and saved `predictions_plot.png`
- [ ] Answered all questions in this writeup with thoughtful responses
- [ ] Pushed all files to GitHub (code, plots, and this writeup)

---

## Optional: Extra Credit (+2 points)

If you want to challenge yourself, modify your code to:
1. Try different train/test split ratios (60/40, 70/30, 90/10)
2. Record the R² score for each split
3. Explain below which split ratio worked best and why you think that is

**YOUR ANSWER:**
