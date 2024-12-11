import gymnasium
from agent import Agent

if __name__ == "__main__":
    env = gymnasium.make("InvertedDoublePendulum-v4", render_mode="human")
    agent = Agent(
        input_dims=env.observation_space.shape,
        env=env,
        num_actions=env.action_space.shape[0],
    )
    n_games = 50000

    best_score = 0
    score_history = []

    f = open("output1.txt", "w")

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0
        while not done:
            # env.render()
            action = agent.select_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            agent.add_to_memory(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_

        if score_history != [] and score > max(score_history):
            agent.save(score)

        score_history.append(score)

        f.write(f"{i}, {score}\n")
        f.flush()
