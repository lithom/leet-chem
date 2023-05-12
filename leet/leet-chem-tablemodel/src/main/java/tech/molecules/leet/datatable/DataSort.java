package tech.molecules.leet.datatable;

public interface DataSort<T>  {
    public String getName();

    /**
     *  1 if a > b
     *  0 if a = b
     * -1 if a < b
     *
     * @param a
     * @param b
     * @return
     */
    public int compare(T a, T b);
}
