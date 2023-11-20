import random
import time


class Particle:
    def __init__(self, n_dimensions, boundaries):
        self.position = []
        self.best_position = []
        self.velocity = []

        for i in range(n_dimensions):
            self.position.append(random.uniform(0, boundaries[i]))
            self.velocity.append(random.uniform(-boundaries[i], boundaries[i]))

        self.best_position = self.position

    def update_position(self, newVal):
        self.position = newVal

    def update_best_position(self, newVal):
        self.best_position = newVal

    def update_velocity(self, newVal):
        self.velocity = newVal


def optimize(
    cost_func,
    n_dimensions,
    boundaries,
    n_particles,
    n_iterations,
    w,
    cp,
    cg,
    verbose=False,
):
    particles = []
    global_solution = []
    gs_eval = []
    gs_history = []
    gs_eval_history = []

    if verbose:
        print("Population initialization...")

    for i in range(n_particles):
        particles.append(Particle(n_dimensions, boundaries))

    global_solution = particles[0].position
    gs_eval = cost_func(global_solution)
    for p in particles:
        p_eval = cost_func(p.best_position)
        if p_eval < gs_eval:
            global_solution = p.best_position
            gs_eval = cost_func(global_solution)

    gs_history.append(global_solution)
    gs_eval_history.append(gs_eval)

    if verbose:
        print("Optimizing")
    start_time = time.time_ns()

    for k in range(n_iterations):
        for p in particles:
            rp = random.uniform(0, 1)
            rg = random.uniform(0, 1)

            velocity = []
            new_position = []
            for i in range(n_dimensions):
                velocity.append(
                    w * p.velocity[i]
                    + cp * rp * (p.best_position[i] - p.position[i])
                    + cg * rg * (global_solution[i] - p.position[i])
                )

                if velocity[i] < -boundaries[i]:
                    velocity[i] = -boundaries[i]
                elif velocity[i] > boundaries[i]:
                    velocity[i] = boundaries[i]

                new_position.append(p.position[i] + velocity[i])
                if new_position[i] < 0.0:
                    new_position[i] = 0.0
                elif new_position[i] > boundaries[i]:
                    new_position[i] = boundaries[i]

            p.update_velocity(velocity)
            p.update_position(new_position)

            p_eval = cost_func(p.position)
            if p_eval < cost_func(p.best_position):
                p.update_best_position(p.position)
                if p_eval < gs_eval:
                    global_solution = p.position
                    gs_eval = p_eval

        gs_eval_history.append(gs_eval)
        gs_history.append(global_solution)

        if verbose:
            printProgressBar(
                k + 1, n_iterations, prefix="Progress:", suffix="Complete", length=50
            )

    finish_time = time.time_ns()
    elapsed_time = (finish_time - start_time) / 10e8

    if verbose:
        time.sleep(0.2)
        print("End of optimization...")
        print()
        print("Optimization elapsed time: {:.2f} s".format(elapsed_time))
        print("Solution evaluation: {:.5f}".format(gs_eval))

    return global_solution, gs_eval, gs_history, gs_eval_history


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
