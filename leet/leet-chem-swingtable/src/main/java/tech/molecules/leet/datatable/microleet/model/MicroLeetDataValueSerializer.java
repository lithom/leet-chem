package tech.molecules.leet.datatable.microleet.model;

public interface MicroLeetDataValueSerializer<C> {

    /**
     * Return null means could not parse
     *
     * @param data
     */
    public C initFromString(String data);
    public String serializeToString(C val);

    public Class getRepresentationClass();
}
