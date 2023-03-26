package tech.molecules.chem.coredb.sql;

import tech.molecules.chem.coredb.Batch;
import tech.molecules.chem.coredb.Tube;

public class TubeImpl implements Tube {
    private String id;
    private Batch batch;

    public TubeImpl(String id, Batch batch) {
        this.id = id;
        this.batch = batch;
    }

    public String getId() { return id; }
    public Batch getBatch() { return batch; }
}
