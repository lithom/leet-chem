package tech.molecules.chem.coredb.sql;

import tech.molecules.chem.coredb.Project;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

public class DBProject {

    public static List<Project> fetchProjects(Connection connection, Set<String> projectIds) throws SQLException {
        List<Project> resultList = new ArrayList<>();

        if (projectIds.isEmpty()) {
            return resultList;
        }

        StringBuilder sqlBuilder = new StringBuilder("SELECT * FROM project WHERE id IN (");
        sqlBuilder.append(String.join(",", Collections.nCopies(projectIds.size(), "?")));
        sqlBuilder.append(")");

        try (PreparedStatement statement = connection.prepareStatement(sqlBuilder.toString())) {
            int index = 1;
            for (String projectId : projectIds) {
                statement.setString(index++, projectId);
            }

            try (ResultSet resultSet = statement.executeQuery()) {
                while (resultSet.next()) {
                    String id = resultSet.getString("id");
                    String name = resultSet.getString("name");
                    resultList.add(new ProjectImpl(id, name));
                }
            }
        }

        return resultList;
    }

}
