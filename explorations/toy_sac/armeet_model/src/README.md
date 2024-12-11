# sac-pendulum
Inverted double pendulum with Soft Actor Critic (SAC) RL model.

## v4
**Highest score**: 9359.85

After 614 episodes (< 5 min of training).

**Demo**:

Here, I've enabled keyboard inputs, which correspond to 0.7 * max_action of left/right input. It's amazing to see how the
actor recovers almost instantly. Of course, for more catastrophic events (like me holding down an arrow key), it is impossible to recover.

https://github.com/user-attachments/assets/8a95bb82-0e4c-4296-a050-814929aa3f20



https://github.com/user-attachments/assets/d70811ca-e7ba-4bd9-a8f8-0a89237f166b

**Score Plots**:
![plot](https://github.com/user-attachments/assets/f575a354-38ec-4680-9cdb-efd275bb23f3)


## v5

I trained the double inverted pendulum using InvertedDoublePendulum-v5 earlier. There was some confusion, and other group members used v4, but I wanted to include my v5 results anyways!

**Highest score**: 82,812

**Demo**: 

https://github.com/user-attachments/assets/0d966f26-0bce-41c8-bf3d-0004cd8b8349

**Score Plots**:
![v5_scores](https://github.com/user-attachments/assets/96e9b5b2-3b32-41ab-aa0c-27d748e505fc)


## Documentation
All documentation is automatically generated by `pdoc3`.  

To generate documentation, run `pdoc --html -o docs . -f`.  

Make sure you do NOT have `pdoc` and only use `pip install pdoc3` or there 
might be package conflicts.

## References
In the code, I sometimes reference back to the original paper + other resources.

> [1] [paper](https://arxiv.org/pdf/1801.01290)  
> [2] [lib](https://skrl.readthedocs.io/en/latest/) - provided by Sorina  
> [3] [article](https://medium.com/@sthanikamsanthosh1994/reinforcement-learning-part-5-soft-actor-critic-sac-network-using-tensorflow2-697917b4b752)  
> [4] [video](https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/SAC/)