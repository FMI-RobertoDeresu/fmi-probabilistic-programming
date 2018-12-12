import pymc as pm

car_door = pm.DiscreteUniform("car_door", lower=1, upper=3)
picked_door = pm.DiscreteUniform("picked_door", lower=1, upper=3)
preference = pm.DiscreteUniform("preference", lower=0, upper=1)


@pm.deterministic
def host_choice(car_door=car_door, picked_door=picked_door, preference=preference):
    if car_door != picked_door: return 6 - car_door - picked_door
    if car_door == 1:
        left = 2
        right = 3
    else:

        left = 1
        if car_door == 2:
            right = 3
        else:
            right = 2
    out = right if preference else left
    return out


@pm.deterministic
def changed_door(picked_door=picked_door, host_choice=host_choice):
    return 6 - host_choice - picked_door


model = pm.Model([car_door, picked_door, preference, host_choice, changed_door])

mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000, 1)
car_door_samples = mcmc.trace('car_door')[:]
picked_door_samples = mcmc.trace('picked_door')[:]
changed_door_samples = mcmc.trace('changed_door')[:]

print()
print()
print("probability to win of a player who stays with the initial choice:",
      (car_door_samples == picked_door_samples).mean())
print("probability to win of a player who switches:",
      (car_door_samples == changed_door_samples).mean())


