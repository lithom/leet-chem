package tech.molecules.chem.coredb.sql;

import tech.molecules.chem.coredb.*;

import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class AssayResultImpl implements AssayResult {
    private long id;
    private Assay assay;
    private Date date;
    private Tube tube;
    private Map<AssayParameter, DataValue> dataValueMap;

    public AssayResultImpl(long id, Assay assay, Date date, Tube tube, Map<AssayParameter, DataValue> dataValueMap) {
        this.id = id;
        this.assay = assay;
        this.date = date;
        this.tube = tube;
        this.dataValueMap = dataValueMap;
    }

    public long getId() { return id; }
    public Assay getAssay() { return assay; }
    public Date getDate() { return date; }
    public Tube getTube() { return tube; }

    public void setAssay(Assay assay) {this.assay = assay;}

    public void setDataValueMap(Map<AssayParameter, DataValue> dataValueMap) { this.dataValueMap = dataValueMap; }

    public DataValue getData(AssayParameter ap) {
        return dataValueMap.get(ap);
    }

    @Override
    public DataValue getData(String parameter_name) {
        List<AssayParameter> pi = this.getAssay().getParameter().stream().filter(ai -> ai.getName().equalsIgnoreCase(parameter_name)).collect(Collectors.toList());
        if(pi.size()==0) {
            System.out.println("[WARN] Parameter "+parameter_name+" not found..");
            return null;
        }
        if(pi.size()>1) {
            System.out.println("[WARN] Multiple assay parameters found for "+parameter_name+" ..");
        }
        return getData(pi.get(0));
    }
}