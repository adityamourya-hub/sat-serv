import urllib.request
import json
import time

repos = [
    "CS-SI/Orekit", "tudat-team/tudatpy", "nyx-space/nyx", "jacobwilliams/Fortran-Astrodynamics-Toolkit",
    "open-space-collective/open-space-toolkit-astrodynamics", "ssmichael1/satkit", "AVSLab/basilisk",
    "esa/pykep", "duncaneddy/brahe", "lox-space/lox", "ATTron/astroz", "IO-Aerospace-software-engineering/Astrodynamics",
    "AlabamaASRL/asset_asrl", "boinor/boinor", "strath-ace/smart-astro", "ssmichael1/astrokit", "ibm/spacetech-ssa",
    "esa/cascade", "ut-astria/caspy", "david-rc-dayton/collision-check", "arma1978/conjunction",
    "RoboticExplorationLab/cvx-spacecraft-cola", "rrovetto/space-situational-awareness-domain-ontology",
    "AshHarvey/ssa-gym", "NTU-ICG/SpaceDet", "cognitiveailab/ssa-corpus"
]

results = {}
for repo in repos:
    url = f"https://api.github.com/repos/{repo}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            results[repo] = {
                "stars": data.get("stargazers_count"),
                "language": data.get("language"),
                "updated_at": data.get("pushed_at") or data.get("updated_at"),
                "license": data.get("license", {}).get("name") if data.get("license") else "None",
            }
    except Exception as e:
        results[repo] = {"error": str(e)}
print(json.dumps(results))
