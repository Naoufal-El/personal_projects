CREATE TABLE doc_logs (
  log_id SERIAL PRIMARY KEY,
  operation VARCHAR(10) NOT NULL,
  doc_id INT NOT NULL,
  doc_type VARCHAR(50),
  doc_author VARCHAR(50),
  log_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


/*Trigger for INSERT operation*/
CREATE OR REPLACE FUNCTION log_insert()
  RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO doc_logs (operation, doc_id, doc_type, doc_author)
  VALUES ('INSERT', NEW.doc_id, NEW.doc_type, NEW.doc_author);
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER after_doc_insert
  AFTER INSERT ON doc
  FOR EACH ROW
  EXECUTE FUNCTION log_insert();



/* Trigger for DELETE operation*/
CREATE OR REPLACE FUNCTION log_delete()
  RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO doc_logs (operation, doc_id, doc_type, doc_author)
  VALUES ('DELETE', OLD.doc_id, OLD.doc_type, OLD.doc_author);
  
  RETURN OLD;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER after_doc_delete
  AFTER DELETE ON doc
  FOR EACH ROW
  EXECUTE FUNCTION log_delete();


/*Trigger for UPDATE operation*/

CREATE OR REPLACE FUNCTION log_update()
  RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO doc_logs (operation, doc_id, doc_type, doc_author)
  VALUES ('UPDATE', NEW.doc_id, NEW.doc_type, NEW.doc_author);
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER after_doc_update
  AFTER UPDATE ON doc
  FOR EACH ROW
  EXECUTE FUNCTION log_update();
