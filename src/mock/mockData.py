corpus = [
    "Dogs are playing on the street",
    "Streets are playing on the dogs",
    "Dogs are playing inside the house",
    "Cats are playing on the street"
]

# TODO: decide if I want to parse also the sinlge ACs (acceptance_criteria entry)
microserviceDataRepresentation = [
    {
        "id": "COMET-1",
        "text": "Dogs are playing on the street",
        # "acceptance_criteria": ["AC1", "AC2", "AC3"],
        "acceptance_criteria": "AC1\nAC2\nAC3",
        "raw_text": "Dogs are playing on the street\n\\Acceptance criterion\n Blablabla\n Note: blablabal"
    },
    {
        "id": "COMET-2",
        "text": "Streets are playing on the dogs",
        "acceptance_criteria": "AC1\nAC2\nAC3",
        "raw_text": "Dogs are playing on the street\n\\Acceptance criterion\n Blablabla\n Note: blablabal"
    },
    {
        "id": "COMET-3",
        "text": "Dogs are playing inside the house",
        "acceptance_criteria": "AC1\nAC2\nAC3",
        "raw_text": "Dogs are playing on the street\n\\Acceptance criterion\n Blablabla\n Note: blablabal"
    },
    {
        "id": "COMET-4",
        "text": "Cats are playing on the street",
        "acceptance_criteria": "AC1\nAC2\nAC3",
        "raw_text": "Dogs are playing on the street\n\\Acceptance criterion\n Blablabla\n Note: blablabal"
    }
]

