import sqlite3


def get_colors(db_name='210920.db'):
    con = sqlite3.connect(db_name)
    cur = con.cursor()
    res = {}
    for row in cur.execute('SELECT music_data_id, circle_type FROM live_data'):
        res[row[0]] = row[1]
    return res

# get_colors()
