from benchmark.schemas import BechmarkQuestion

# Multi-Hazard Questions
data_single_hazard = {
    "agricultural drought": {
        "Angola": 4734925,
        # "Nicaragua": 1025854,
        # "Uruguay": 261217,
        # "Colombia": 3653606,
    },
    "air pollution": {
        # "Angola": 18705698,
        # "Nicaragua": 2311490,
        # "Uruguay": 664396,
        "Colombia": 12998935,
    },
    "coastal floods": {
        # "Angola": 4561,
        # "Nicaragua": 7392,
        "Uruguay": 1099,
        # "Colombia": 29475,
    },
    "drought SPEI": {
        # "Angola": 637076,
        "Nicaragua": 12040,
        # "Uruguay": 139014,
        # "Colombia": 2769387,
    },
    "drought SPI": {
        # "Angola": 99655,
        "Nicaragua": 12040,
        # "Uruguay": 134579,
        # "Colombia": 1691547,
    },
    "extreme heat": {
        # "Angola": 2751854,
        # "Nicaragua": 672663,
        # "Uruguay": 0,
        "Colombia": 1489703,
    },
    "fire frequency": {
        "Angola": 6238879,
        # "Nicaragua": 104389,
        # "Uruguay": 140182,
        # "Colombia": 187259,
    },
    "fire intensity": {
        # "Angola": 1347117,
        # "Nicaragua": 86302,
        "Uruguay": 68997,
        # "Colombia": 690200,
    },
    "heatwave duration": {
        # "Angola": 16557210,
        # "Nicaragua": 2265486,
        # "Uruguay": 0,
        "Colombia": 10830410,
    },
    "heatwave frequency": {
        "Angola": 15860450,
        # "Nicaragua": 2145364,
        # "Uruguay": 7754,
        # "Colombia": 12180450,
    },
    "heatwave severity": {
        # "Angola": 17,
        # "Nicaragua": 0,
        "Uruguay": 410675,
        # "Colombia": 37066,
    },
    "river floods": {
        "Angola": 661223,
        # "Nicaragua": 47062,
        # "Uruguay": 49735,
        # "Colombia": 930693,
    },
    "sand and dust storms": {
        # "Angola": 1171088,
        # "Nicaragua": 43,
        # "Uruguay": 70,
        "Colombia": 36200,
    },
    "tropical storms": {
        # "Angola": 0,
        # "Nicaragua": 2382957,
        "Uruguay": 0,
        # "Colombia": 2009752,
    },
    "vectorborne malaria pv": {
        # "Angola": 0,
        "Nicaragua": 534003,
        # "Uruguay": 0,
        # "Colombia": 7155117,
    },
    "vectorborne malaria pf": {
        "Angola": 18325996,
        # "Nicaragua": 90363,
        # "Uruguay": 0,
        # "Colombia": 432189,
    },
}
data_multi_hazard = {
    # and river and coastal floods
    "river and coastal floods": {
        "Colombia": 16638,
        # "Angola": 1347,
        # "Nicaragua": 3204,
        # "Uruguay": 937,
    },
    # or river or coastal floods
    "river or coastal floods": {
        # "Colombia": 943530,
        # "Angola": 664436,
        # "Nicaragua": 51250,
        "Uruguay": 49897,
    },
    # and malaria
    "both kinds of malaria": {
        # "Colombia": 432180,
        "Angola": 0,
        # "Nicaragua": 52274,
        # "Uruguay": 0,
    },
    # or malaria
    "any kind of malaria": {
        # "Colombia": 7155125,
        # "Angola": 18325995,
        "Nicaragua": 572093,
        # "Uruguay": 0,
    },
}


# Iterate through data to create benchmark questions
# Combine single and multi-hazard data
data = {**data_single_hazard, **data_multi_hazard}
benchmark_questions: list[BechmarkQuestion] = []
for hazard_name, countries in data.items():
    for country, value in countries.items():
        benchmark_questions.append(
            BechmarkQuestion(
                question=f"How many children were exposed to {hazard_name} in {country}",
                answer=value,
                response_type="numerical",
            )
        )
