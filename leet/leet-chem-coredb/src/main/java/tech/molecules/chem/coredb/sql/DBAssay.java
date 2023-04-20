package tech.molecules.chem.coredb.sql;

import tech.molecules.chem.coredb.Assay;
import tech.molecules.chem.coredb.AssayParameter;
import tech.molecules.chem.coredb.DataType;
import tech.molecules.chem.coredb.Project;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.*;

public class DBAssay {

    public static List<Assay> fetchAssays(Connection connection, Set<Integer> ids) throws SQLException {
        if (ids == null || ids.isEmpty()) {
            return Collections.emptyList();
        }

        StringBuilder queryBuilder = new StringBuilder(
                "SELECT assay.id AS assay_id, assay.name AS assay_name, assay.project_id AS assay_project_id , " +
                        "assay_parameter.name AS parameter_name, assay_parameter.id AS assay_parameter_id, " +
                        "data_type.name AS data_type_name , data_type.id AS data_type_id " +
                        "FROM assay " +
                        "JOIN assay_parameter ON assay.id = assay_parameter.assay_id " +
                        "JOIN data_type ON assay_parameter.data_type_id = data_type.id " +
                        "WHERE assay.id IN ("
        );

        StringJoiner idPlaceholders = new StringJoiner(", ");
        for (Integer id : ids) {
            idPlaceholders.add("?");
        }
        queryBuilder.append(idPlaceholders);
        queryBuilder.append(") ORDER BY assay.id");

        PreparedStatement statement = connection.prepareStatement(queryBuilder.toString());
        int i = 1;
        for (Integer id : ids) {
            statement.setInt(i++, id);
        }

        ResultSet resultSet = statement.executeQuery();
        Map<Integer, AssayImpl> assayMap = new LinkedHashMap<>();
        Map<Integer, String> assayToProjectMap = new HashMap<>();
        // fetch projects..
        Map<String,Project> projects = new HashMap<>();
        //List<Project> projects_list = DBProject.fetchProjects(connection, new HashSet<>(assayToProjectMap.values()) );
        //projects_list.stream().forEach(pi -> projects.put(pi.getId(),pi));


        while (resultSet.next()) {
            int assayId = resultSet.getInt("assay_id");
            AssayImpl assay = assayMap.get(assayId);
            if (assay == null) {
                String assayName = resultSet.getString("assay_name");
                String assayProject = resultSet.getString("assay_project_id");
                if(!projects.containsKey(assayProject)) {
                    Project pi = DBProject.fetchProjects(connection,Collections.singletonMap(assayProject,"").keySet()).get(0);
                    projects.put(pi.getId(),pi);
                }
                assay = new AssayImpl(assayId, assayName,projects.get(assayProject), new ArrayList<>());
                assayMap.put(assayId, assay);
                assayToProjectMap.put(assayId,assayProject);
            }

            String parameterName = resultSet.getString("parameter_name");
            String dataTypeName = resultSet.getString("data_type_name");
            int dataTypeId = resultSet.getInt("data_type_id");
            int assayParameterId = resultSet.getInt("assay_parameter_id");
            DataType dataType = new DataTypeImpl(dataTypeId,dataTypeName);
            AssayParameter parameter = new AssayParameterImpl(assayParameterId, assay, dataType, parameterName);
            assay.getParameter().add(parameter); // You need to add a method "addParameter" to the AssayImpl class.
        }

        return new ArrayList<>(assayMap.values());
    }

}
