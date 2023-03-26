package tech.molecules.chem.coredb.sql;

import tech.molecules.chem.coredb.Batch;
import tech.molecules.chem.coredb.Compound;

public class BatchImpl implements Batch {
    private String id;
    private Compound compound;

    public BatchImpl(String id, Compound compound) {
        this.id = id;
        this.compound = compound;
    }

    public String getId() { return id; }
    public Compound getCompound() { return compound; }
}
