package tech.molecules.chem.coredb.sql;

import com.actelion.research.chem.StereoMolecule;
import tech.molecules.chem.coredb.*;
import tech.molecules.leet.chem.ChemUtils;

import java.sql.*;
import java.sql.Date;
import java.util.*;

public abstract class DBManager implements CoreDB, CoreDBWriter {

    private SQLHelper helper;
    private Connection connection;

    public DBManager(Connection connection, SQLHelper helper) {
        this.connection = connection;
        this.helper = helper;
    }

    @Override
    public List<Assay> fetchAssays(Set<Integer> ids) throws Exception {
        return DBAssay.fetchAssays(connection,ids);
    }

    @Override
    public List<AssayResult> searchAssayResults(AssayResultQuery query) throws Exception {
        return DBAssayResult.searchAssayResults(connection,query);
    }

    @Override
    public List<Project> fetchProjects(Set<String> projectIds) throws SQLException {
        return DBProject.fetchProjects(connection,projectIds);
    }





    @Override
    public Connection getConnection() {
        return this.connection;
    }

    @Override
    public Project createProject(String id, String name) throws SQLException {
        //PreparedStatement statement = connection.prepareStatement("INSERT or IGNORE INTO project (id, name) VALUES (?, ?)");
        PreparedStatement statement = connection.prepareStatement(helper.getInsertOrIgnoreStatement("project","id, name","?, ?"));
        statement.setString(1, id);
        statement.setString(2, name);
        statement.executeUpdate();
        return new ProjectImpl(id, name);
    }

    @Override
    public DataType createDataType(String name) throws SQLException {
        //String query = "INSERT INTO data_type (name) VALUES (?)";
        String query = helper.getInsertOrIgnoreStatement("data_type","name","?");
        PreparedStatement statement = connection.prepareStatement(query, Statement.RETURN_GENERATED_KEYS);
        statement.setString(1, name);

        statement.executeUpdate();
        ResultSet resultSet = statement.getGeneratedKeys();
        if (resultSet.next()) {
            int id = resultSet.getInt(1);
            return new DataTypeImpl(id, name);
        } else {
            throw new SQLException("Failed to create data type");
        }
    }

    @Override
    public Assay createAssay(String name, Project project) throws SQLException {
        String query = "INSERT INTO assay (name, project_id) VALUES (?, ?)";
        PreparedStatement statement = connection.prepareStatement(query, Statement.RETURN_GENERATED_KEYS);
        statement.setString(1, name);
        statement.setString(2, project.getId());

        statement.executeUpdate();
        ResultSet resultSet = statement.getGeneratedKeys();
        if (resultSet.next()) {
            int id = resultSet.getInt(1);
            return new AssayImpl(id, name, project, new ArrayList<>() );
        } else {
            throw new SQLException("Failed to create assay");
        }
    }

    @Override
    public AssayParameter createAssayParameter(Assay assay, DataType dataType, String name) throws SQLException {
        String query = "INSERT INTO assay_parameter (assay_id, data_type_id, name) VALUES (?, ?, ?)";
        PreparedStatement statement = connection.prepareStatement(query, Statement.RETURN_GENERATED_KEYS);
        statement.setInt(1, assay.getId());
        statement.setInt(2, dataType.getId());
        statement.setString(3, name);

        statement.executeUpdate();
        ResultSet resultSet = statement.getGeneratedKeys();


        if (resultSet.next()) {
            int id = resultSet.getInt(1);
            return new AssayParameterImpl(id, assay, dataType, name);
        } else {
            throw new SQLException("Failed to create assay parameter");
        }

    }

    @Override
    public AssayResult createAssayResult(Assay assay, java.util.Date date, Tube tube) throws SQLException {
        String query = "INSERT INTO assay_result (assay_id, date, tube_id) VALUES (?, ?, ?)";
        PreparedStatement statement = connection.prepareStatement(query, Statement.RETURN_GENERATED_KEYS);
        statement.setInt(1, assay.getId());
        statement.setDate(2, new java.sql.Date(date.getTime()));
        statement.setString(3, tube.getId());

        statement.executeUpdate();
        ResultSet resultSet = statement.getGeneratedKeys();
        if (resultSet.next()) {
            long id = resultSet.getLong(1);
            return new AssayResultImpl(id, assay, date, tube, new HashMap<>());
        } else {
            throw new SQLException("Failed to create assay result");
        }
    }

    @Override
    public Compound createCompound(String id, StereoMolecule molecule) throws SQLException, CoreDBException {
        if(molecule == null) {
            throw new CoreDBException("[createCompound] StereoMolecule is null, id = "+id);
        }
        PreparedStatement statement = connection.prepareStatement("INSERT INTO compound (id, idcode, idcode_coordinates) VALUES (?,?,?)");
        statement.setString(1, id);
        statement.setString(2, molecule.getIDCode());
        statement.setString(3, molecule.getIDCoordinates());
        statement.executeUpdate();
        return new CompoundImpl(id, molecule);
    }

    @Override
    public Batch createBatch(String id, Compound compound) throws SQLException {
        PreparedStatement statement = connection.prepareStatement("INSERT INTO batch (id, compound_id) VALUES (?, ?)");
        statement.setString(1, id);
        statement.setString(2, compound.getId());
        statement.executeUpdate();
        return new BatchImpl(id, compound);
    }

    @Override
    public Tube createTube(String id, Batch batch) throws SQLException {
        PreparedStatement statement = connection.prepareStatement("INSERT INTO tube (id, batch_id) VALUES (?, ?)");
        statement.setString(1, id);
        statement.setString(2, batch.getId());
        statement.executeUpdate();
        return new TubeImpl(id, batch);
    }

    @Override
    public void addDataValue(AssayResult assayResult, AssayParameter assayParameter, DataValue dataValue) throws SQLException {
        String query = "INSERT INTO assay_result_data (assay_result_id, assay_parameter_id, double_value, text_value) VALUES (?, ?, ?, ?)";
        PreparedStatement statement = connection.prepareStatement(query);
        statement.setLong(1, assayResult.getId());
        statement.setInt(2, assayParameter.getId());

        statement.setDouble(3, dataValue.getAsDouble());
        statement.setString(4, dataValue.getAsText());

        int affectedRows = statement.executeUpdate();
        if (affectedRows == 0) {
            throw new SQLException("Failed to add data value");
        }
    }




    @Override
    public List<Tube> searchTubes(TubeQuery query) throws SQLException {
        List<Tube> result = new ArrayList<>();
        StringBuilder sb = new StringBuilder("SELECT tube.id as tube_id, tube.batch_id, batch.compound_id, compound.idcode ")
                .append("FROM tube ")
                .append("JOIN batch ON tube.batch_id = batch.id ")
                .append("JOIN compound ON batch.compound_id = compound.id ")
                .append("WHERE 1=1");

        if (query.getBatchId() != null) {
            sb.append(" AND tube.batch_id = ?");
        }

        PreparedStatement statement = connection.prepareStatement(sb.toString());
        int parameterIndex = 1;

        if (query.getBatchId() != null) {
            statement.setString(parameterIndex, query.getBatchId());
        }

        ResultSet resultSet = statement.executeQuery();
        while (resultSet.next()) {
            String tubeId = resultSet.getString("tube_id");
            String batchId = resultSet.getString("batch_id");
            String compoundId = resultSet.getString("compound_id");
            String compoundIdcode = resultSet.getString("idcode");

            StereoMolecule molecule = ChemUtils.parseIDCode(compoundIdcode);
            Compound compound = new CompoundImpl(compoundId, molecule);
            Batch batch = new BatchImpl(batchId, compound);

            result.add(new TubeImpl(tubeId, batch));
        }

        return result;
    }


    @Override
    public List<Project> searchProjects(ProjectQuery query) throws SQLException {
        List<Project> result = new ArrayList<>();
        StringBuilder sb = new StringBuilder("SELECT * FROM project WHERE 1=1");

        if (query.getName() != null) {
            sb.append(" AND name LIKE ?");
        }

        PreparedStatement statement = connection.prepareStatement(sb.toString());
        int parameterIndex = 1;

        if (query.getName() != null) {
            statement.setString(parameterIndex, "%" + query.getName() + "%");
        }

        ResultSet resultSet = statement.executeQuery();
        while (resultSet.next()) {
            String id = resultSet.getString("id");
            String name = resultSet.getString("name");

            result.add(new ProjectImpl(id, name));
        }

        return result;
    }

    @Override
    public List<Assay> searchAssays(AssayQuery query) throws SQLException {
        Set<Integer> assayIds = new HashSet<>();

        StringBuilder sqlBuilder = new StringBuilder("SELECT id FROM assay");

        if (query.getProjectId() != null) {
            sqlBuilder.append(" WHERE project_id = ?");
        }

        try (PreparedStatement statement = connection.prepareStatement(sqlBuilder.toString())) {
            if (query.getProjectId() != null) {
                statement.setString(1, query.getProjectId());
            }

            try (ResultSet resultSet = statement.executeQuery()) {
                while (resultSet.next()) {
                    int id = resultSet.getInt("id");
                    assayIds.add(id);
                }
            }
        }

        return DBAssay.fetchAssays(connection, assayIds);
    }



    @Override
    public int getNumberOfMeasurements(Assay assay) throws SQLException {
        PreparedStatement statement = connection.prepareStatement(
                "SELECT COUNT(*) FROM assay_result WHERE assay_id = ?"
        );
        statement.setInt(1, assay.getId());
        ResultSet resultSet = statement.executeQuery();

        if (resultSet.next()) {
            return resultSet.getInt(1);
        } else {
            return 0;
        }
    }

}


