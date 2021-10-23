from .academy_counterattack_easy import Academy_Counterattack_Easy
from .academy_counterattack_hard import Academy_Counterattack_Hard
from .five_vs_five import Five_vs_Five

REGISTRY = {}
REGISTRY["academy_counterattack_easy"] = Academy_Counterattack_Easy
REGISTRY["academy_counterattack_hard"] = Academy_Counterattack_Hard
REGISTRY["five_vs_five"] = Five_vs_Five

'''
Google Football
'''
def GoogleFootballEnv(**kwargs): 
    map_name = kwargs["map_name"]
    if map_name in REGISTRY:
        return REGISTRY[map_name]()
    else:
        raise "Google Football: map is not implementated!"
