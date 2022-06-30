voice_option = {
    'name': 'Voice_QoE',
    'seq_len': 173,
    # 'seq_len': 692,
    # 'seq_len': 2249,
    'file_path': './dataset/voice_4domain_processed.csv',
    # 'file_path': './dataset/test/cali_test.csv',

    'input_dim': 18,
    'learning_rate': 0.0001,
    'weight_decay': 0.001,

    'domains': ['amelkot', 'awekesa', 'aturaga', 'avanausdale', 'aosburn', 'ctran', 'drajaneni', 'dwhite', 'dwarrier',
                'dpatel', 'ealvarez', 'frabbi', 'gteja', 'gnagaraj', 'jwologo', 'jlopez', 'jsoriano', 'jbui', 'krajpal',
                'krashid', 'ksrinivas', 'kpatel', 'lchellappan', 'lpericharla', 'lcalder', 'manguyen', 'minguyen',
                'mguzman', 'marshad', 'nshrestha', 'ncamacho', 'rcharan', 'schecka', 'svalllivedu', 'smanchikanti',
                'salta', 'skakarla', 'ttomlinson', 'wwilson', 'ygali'],

    'domain_to_number': {"amelkot": 0, "awekesa": 1, "aturaga": 2, "avanausdale": 3, "aosburn": 4, "ctran": 5,
                         "drajaneni": 6, "dwhite": 7, "dwarrier": 8, "dpatel": 9, "ealvarez": 10, "frabbi": 11,
                         "gteja": 12, "gnagaraj": 13, "jwologo": 14, "jlopez": 15, "jsoriano": 16, "jbui": 17,
                         "krajpal": 18, "krashid": 19, "ksrinivas": 20, "kpatel": 21, "lchellappan": 22,
                         "lpericharla": 23, "lcalder": 24, "manguyen": 25, "minguyen": 26, "mguzman": 27,
                         "marshad": 28, "nshrestha": 29, "ncamacho": 30, "rcharan": 31, "schecka": 32, "svallivedu": 33,
                         "smanchikanti": 34, "salta": 35, "skakarla": 36, "ttomlinson": 37, "wwilson": 38, "ygail": 39},

    'number_to_domain': {0: "amelkot", 1: "awekesa", 2: "aturaga", 3: "avanausdale", 4: "aosburn", 5: "ctran",
                         6: "drajaneni", 7: "dwhite", 8: "dwarrier", 9: "dpatel", 10: "ealvarez", 11: "frabbi",
                         12: "gteja", 13: "gnagaraj", 14: "jwologo", 15: "jlopez", 16: "jsoriano", 17: "jbui",
                         18: "krajpal", 19: "krashid", 20: "ksrinivas", 21: "kpatel", 22: "lchellappan",
                         23: "lpericharla", 24: "lcalder", 25: "manguyen", 26: "minguyen", 27: "mguzman",
                         28: "marshad", 29: "nshrestha", 30: "ncamacho", 31: "rcharan", 32: "schecka", 33: "svallivedu",
                         34: "smanchikanti", 35: "salta", 36: "skakarla", 37: "ttomlinson", 38: "wwilson", 39: "ygail"},

    'classes': ['1', '2', '3', '4', '5'],

    'num_class': 5
}
