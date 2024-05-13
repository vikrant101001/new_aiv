def insert_conversation(user_question, bot_answer, careteam_id, caregiver_id):
  try:
    # Create a SQLAlchemy engine and session
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create a Conversation object
    conversation = advocatehistory(user_question=user_question,
                                   bot_answer=bot_answer,
                                   careteam_id=careteam_id,
                                   caregiver_id=caregiver_id)

    # Add the Conversation object to the session and commit the transaction
    session.add(conversation)
    session.commit()

    # Close the session
    session.close()

  except Exception as e:
    # Handle exceptions (e.g., database errors)
    print(f"Error inserting conversation: {e}")


def insert_checkin(checkin_question, user_answer, caregiver_id, careteam_id):
  try:
    # Create a SQLAlchemy engine and session
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create a Conversation object
    conversation = checkinhistory(checkin_question=checkin_question,
                                  user_answer=user_answer,
                                  caregiver_id=caregiver_id,
                                  careteam_id=careteam_id)

    # Add the Conversation object to the session and commit the transaction
    session.add(conversation)
    session.commit()

    # Close the session
    session.close()

  except Exception as e:
    # Handle exceptions (e.g., database errors)
    print(f"Error inserting conversation: {e}")