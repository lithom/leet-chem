package tech.molecules.leet.chem.injector.gui;

import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.gui.JEditableStructureView;
import com.actelion.research.gui.editor.SwingEditorPanel;
import tech.molecules.leet.table.gui.JExtendedEditorPanel;

import javax.swing.*;
import java.awt.*;

public class JInjectorMainPanel extends JPanel {

    private JSplitPane jsp_Main;
    private JSplitPane jsp_Main_Left;

    private JPanel jp_Main_Editor;
    private JPanel jp_Main_EditorTop;
    private JPanel jp_Main_Lower;

    private JExtendedEditorPanel editor;

    //private JMenuBar jmb_editor;


    public JInjectorMainPanel() {
        init();
    }

    private void init() {
        this.removeAll();
        this.setLayout(new BorderLayout());

        this.jsp_Main = new JSplitPane();
        this.jsp_Main_Left = new JSplitPane(JSplitPane.VERTICAL_SPLIT);

        this.jsp_Main.setLeftComponent(this.jsp_Main_Left);

        this.jp_Main_Editor = new JPanel();
        this.jsp_Main_Left.setTopComponent(jp_Main_Editor);

        this.jp_Main_Lower = new JPanel();
        this.jsp_Main_Left.setBottomComponent(this.jp_Main_Lower);

        editor = new JExtendedEditorPanel();
        this.jp_Main_Editor.setLayout(new BorderLayout());
        this.jp_Main_Editor.add(editor,BorderLayout.CENTER);

        //this.jp_Main_EditorTop = new JPanel();
        //this.jmb_editor = new JMenuBar();
        //this.jp_Main_EditorTop.add(jmb_editor);

        initEditorMenuBar();

        this.add(this.jsp_Main,BorderLayout.CENTER);
    }

    private void initEditorMenuBar() {
        JMenu jm_injector = new JMenu("Injector");
        JMenuItem jmi_proposeStructures = new JMenuItem("Propose structures for selection");
        jm_injector.add(jmi_proposeStructures);
        this.editor.getMenuBar().add(jm_injector);
    }



}
