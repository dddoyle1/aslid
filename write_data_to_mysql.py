from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    BigInteger,
    String,
    ForeignKey,
    Float,
    MetaData,
)
from sqlalchemy.orm import declarative_base, Session, relationship

import argparse

from getpass import getpass

import pandas as pd
import numpy as np
import json
from tqdm import tqdm

from aslid.data import get_training_data_paths
import aslid.model

parser = argparse.ArgumentParser()
parser.add_argument("--recreate", action="store_true")
args = parser.parse_args()

# load participant metadata
(
    input_files,
    sign_indices,
    participant_ids,
    sequence_ids,
    sign_labels,
) = get_training_data_paths(
    "data/train.csv",
    "data/sign_to_prediction_index_map.json",
    rootdir="data/",
    return_ids=True,
)

# determine required length of string column
with open("data/sign_to_prediction_index_map.json", "r") as j:
    prediction_map = json.load(j)

max_sign_length = len(max(list(prediction_map.keys()), key=len)) + 3

# Connect to the database
host = "192.168.0.9"
db_name = "aslid_data"
user = "wsl"
pw = getpass()

engine = create_engine(f"mysql+pymysql://{user}:{pw}@{host}/{db_name}")


if args.recreate:
    metadata = MetaData()
    metadata.reflect(bind=engine)
    metadata.drop_all(engine)

# declare table and row structure
Base = declarative_base()


class Participant(Base):
    __tablename__ = "participant"
    id = Column(Integer, primary_key=True, nullable=False)

    sequence = relationship("Sequence", back_populates="participant")

    landmark_coordinates = relationship(
        "LandmarkCoordinates", back_populates="participant"
    )


class Sequence(Base):
    __tablename__ = "sequence"
    id = Column(BigInteger, primary_key=True)

    participant_id = Column(Integer, ForeignKey("participant.id"), nullable=False)
    participant = relationship("Participant", back_populates="sequence")

    sign_index = Column(Integer, ForeignKey("sign.id"), nullable=True)
    sign = relationship("Sign", back_populates="sequence")

    landmark_coordinates = relationship(
        "LandmarkCoordinates", back_populates="sequence"
    )

    def __init__(self, id, participant_id, sign_index):
        self.id = id
        self.participant_id = participant_id
        self.sign_index = aslid.model.SIGN_ID_OFFSET + sign_index


class Sign(Base):
    __tablename__ = "sign"
    id = Column(Integer, nullable=False, primary_key=True, autoincrement=True)
    label = Column(String(length=max_sign_length), nullable=False)
    sequence = relationship("Sequence", back_populates="sign")

    def __init__(self, id, label):
        self.id = id + aslid.model.SIGN_ID_OFFSET
        self.label = label


class LandmarkCoordinates(Base):
    __tablename__ = "landmark_coordinates"
    id = Column(Integer, primary_key=True, autoincrement=True)

    participant_id = Column(Integer, ForeignKey("participant.id"), nullable=False)
    participant = relationship("Participant", back_populates="landmark_coordinates")

    sequence_id = Column(BigInteger, ForeignKey("sequence.id"), nullable=False)
    sequence = relationship("Sequence", back_populates="landmark_coordinates")

    frame = Column(Integer, nullable=False)

    landmark_type = Column(String(length=10), nullable=False, primary_key=True)
    landmark_index = Column(Integer, nullable=False, primary_key=True)

    x = Column(Float, nullable=True)
    y = Column(Float, nullable=True)
    z = Column(Float, nullable=True)

    def __init__(self, row, participant_id, sequence_id):
        self.participant_id = participant_id
        self.sequence_id = sequence_id
        self.frame = row["frame"]
        self.landmark_type = row["type"]
        self.landmark_index = row["landmark_index"]
        self.x = None if pd.isna(row["x"]) else row["x"]
        self.y = None if pd.isna(row["y"]) else row["y"]
        self.z = None if pd.isna(row["z"]) else row["z"]


# create tables
Base.metadata.create_all(engine)

participants = [Participant(id=pid) for pid in np.unique(participant_ids)]

# I guess SQL doesn't like primary key id of zero. Offset by 1000 to avoid mistakes in parsing later
signs = [Sign(id=idx, label=label) for label, idx in prediction_map.items()]

# associate sequences with participants
sequences = [
    Sequence(id=sid, participant_id=pid, sign_index=sign_index)
    for sid, pid, sign_index in zip(sequence_ids, participant_ids, sign_indices)
]

commit_every = 50
# write participants and sequence tables
with Session(engine) as session:
    session.add_all(participants)
    session.add_all(signs)
    session.add_all(sequences)

    for i in tqdm(range(len(input_files))):
        df = pd.read_parquet(input_files[i])
        session.add_all(
            df.apply(
                LandmarkCoordinates,
                participant_id=participant_ids[i],
                sequence_id=sequence_ids[i],
                axis=1,
            ).values
        )
        if i % commit_every == 0:
            session.commit()

    # session.add_all(list(participants.values()))
    session.commit()

# db.execute(create_participants_table)

# create sequences table


# df = pd.read_parquet(input_file)
# df["participant_id"] = participant_id
# df["sequence_id"] = sequence_id
# # df = df.set_index("row_id")

# df.to_sql(f"{participant_id}_{sequence_id}", con=db, if_exists="replace")
