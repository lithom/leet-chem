package tech.molecules.leet.gui.chem.project;

import tech.molecules.leet.commons.tree.AbstractMappedTreeNode;
import tech.molecules.leet.gui.chem.project.action.ObjectSpecific;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.util.*;

public class ProjectTreeController {

    private final Map<Class<?>, List<Action>> nodeTypeActions = new HashMap<>();
    private List<Class<?>> checkedClasses = new ArrayList<>();

    public void setNodeTypeActions(Class<?> nodeType, List<Action> actions) {
        nodeTypeActions.put(nodeType, actions);
    }

    public void setListOfCheckedClasses(List<Class<?>> classes) {
        this.checkedClasses = classes;
    }

    public JPopupMenu createContextMenu(AbstractMappedTreeNode<?> node) {
        JPopupMenu contextMenu = new JPopupMenu();
        Object nodeObject = node.getNodeObject();

        for (Class<?> checkedClass : checkedClasses) {
            if (checkedClass.isInstance(nodeObject)) {
                List<Action> actions = nodeTypeActions.get(checkedClass);
                if (actions != null) {
                    for (Action action : actions) {
                        if(action instanceof ObjectSpecific) {
                            try{
                                ((ObjectSpecific)action).setObject(nodeObject);
                            }
                            catch(Exception ex){}
                        }
                        contextMenu.add(action);
                    }
                }
            }
        }

        return contextMenu.getComponentCount() > 0 ? contextMenu : null;
    }
}

