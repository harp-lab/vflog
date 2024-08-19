
#include "vflog.cuh"

#include <iostream>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

void lubm(char *data_path) {
    auto global_buffer = std::make_shared<vflog::d_buffer>(40960);

    KernelTimer timer;
    timer.start_timer();
    // input relations
    std::string src_advisor_file =
        std::string(data_path) + "/src_advisor.facts";
    vflog::multi_hisa src_advisor(2, src_advisor_file.c_str(), global_buffer);
    std::string src_AssistantProfessor_file =
        std::string(data_path) + "/src_AssistantProfessor.facts";
    vflog::multi_hisa src_AssistantProfessor(
        1, src_AssistantProfessor_file.c_str(), global_buffer);
    std::string src_AssociateProfessor_file =
        std::string(data_path) + "/src_AssociateProfessor.facts";
    vflog::multi_hisa src_AssociateProfessor(
        1, src_AssociateProfessor_file.c_str(), global_buffer);
    std::string src_Course_file = std::string(data_path) + "/src_Course.facts";
    vflog::multi_hisa src_Course(1, src_Course_file.c_str(), global_buffer);
    std::string src_Department_file =
        std::string(data_path) + "/src_Department.facts";
    vflog::multi_hisa src_Department(1, src_Department_file.c_str(),
                                     global_buffer);
    std::string src_doctoralDegreeFrom_file =
        std::string(data_path) + "/src_doctoralDegreeFrom.facts";
    vflog::multi_hisa src_doctoralDegreeFrom(
        2, src_doctoralDegreeFrom_file.c_str(), global_buffer);
    std::string src_emailAddress_file =
        std::string(data_path) + "/src_emailAddress.facts";
    vflog::multi_hisa src_emailAddress(2, src_emailAddress_file.c_str(),
                                       global_buffer);
    std::string src_FullProfessor_file =
        std::string(data_path) + "/src_FullProfessor.facts";
    vflog::multi_hisa src_FullProfessor(1, src_FullProfessor_file.c_str(),
                                        global_buffer);
    std::string src_GraduateCourse_file =
        std::string(data_path) + "/src_GraduateCourse.facts";
    vflog::multi_hisa src_GraduateCourse(1, src_GraduateCourse_file.c_str(),
                                         global_buffer);
    std::string src_GraduateStudent_file =
        std::string(data_path) + "/src_GraduateStudent.facts";
    vflog::multi_hisa src_GraduateStudent(1, src_GraduateStudent_file.c_str(),
                                          global_buffer);
    std::string src_headOf_file = std::string(data_path) + "/src_headOf.facts";
    vflog::multi_hisa src_headOf(2, src_headOf_file.c_str(), global_buffer);
    std::string src_Lecturer_file =
        std::string(data_path) + "/src_Lecturer.facts";
    vflog::multi_hisa src_Lecturer(1, src_Lecturer_file.c_str(), global_buffer);
    std::string src_mastersDegreeFrom_file =
        std::string(data_path) + "/src_mastersDegreeFrom.facts";
    vflog::multi_hisa src_mastersDegreeFrom(
        2, src_mastersDegreeFrom_file.c_str(), global_buffer);
    std::string src_memberOf_file =
        std::string(data_path) + "/src_memberOf.facts";
    vflog::multi_hisa src_memberOf(2, src_memberOf_file.c_str(), global_buffer);
    std::string src_name_file = std::string(data_path) + "/src_name.facts";
    vflog::multi_hisa src_name(2, src_name_file.c_str(), global_buffer);
    std::string src_Publication_file =
        std::string(data_path) + "/src_Publication.facts";
    vflog::multi_hisa src_Publication(1, src_Publication_file.c_str(),
                                      global_buffer);
    std::string src_publicationAuthor_file =
        std::string(data_path) + "/src_publicationAuthor.facts";
    vflog::multi_hisa src_publicationAuthor(
        2, src_publicationAuthor_file.c_str(), global_buffer);
    std::string src_ResearchAssistant_file =
        std::string(data_path) + "/src_ResearchAssistant.facts";
    vflog::multi_hisa src_ResearchAssistant(
        1, src_ResearchAssistant_file.c_str(), global_buffer);
    std::string src_ResearchGroup_file =
        std::string(data_path) + "/src_ResearchGroup.facts";
    vflog::multi_hisa src_ResearchGroup(1, src_ResearchGroup_file.c_str(),
                                        global_buffer);
    std::string src_researchInterest_file =
        std::string(data_path) + "/src_researchInterest.facts";
    vflog::multi_hisa src_researchInterest(2, src_researchInterest_file.c_str(),
                                           global_buffer);
    std::string src_subOrganizationOf_file =
        std::string(data_path) + "/src_subOrganizationOf.facts";
    vflog::multi_hisa src_subOrganizationOf(
        2, src_subOrganizationOf_file.c_str(), global_buffer, 1);
    std::string src_takesCourse_file =
        std::string(data_path) + "/src_takesCourse.facts";
    vflog::multi_hisa src_takesCourse(2, src_takesCourse_file.c_str(),
                                      global_buffer);
    std::string src_teacherOf_file =
        std::string(data_path) + "/src_teacherOf.facts";
    vflog::multi_hisa src_teacherOf(2, src_teacherOf_file.c_str(),
                                    global_buffer);
    std::string src_TeachingAssistant_file =
        std::string(data_path) + "/src_TeachingAssistant.facts";
    vflog::multi_hisa src_TeachingAssistant(
        1, src_TeachingAssistant_file.c_str(), global_buffer);
    std::string src_teachingAssistantOf_file =
        std::string(data_path) + "/src_teachingAssistantOf.facts";
    vflog::multi_hisa src_teachingAssistantOf(
        2, src_teachingAssistantOf_file.c_str(), global_buffer);
    std::string src_telephone_file =
        std::string(data_path) + "/src_telephone.facts";
    vflog::multi_hisa src_telephone(2, src_telephone_file.c_str(),
                                    global_buffer);
    std::string src_undergraduateDegreeFrom_file =
        std::string(data_path) + "/src_undergraduateDegreeFrom.facts";
    vflog::multi_hisa src_undergraduateDegreeFrom(
        2, src_undergraduateDegreeFrom_file.c_str(), global_buffer);
    std::string src_UndergraduateStudent_file =
        std::string(data_path) + "/src_UndergraduateStudent.facts";
    vflog::multi_hisa src_UndergraduateStudent(
        1, src_UndergraduateStudent_file.c_str(), global_buffer);
    std::string src_University_file =
        std::string(data_path) + "/src_University.facts";
    vflog::multi_hisa src_University(1, src_University_file.c_str(),
                                     global_buffer);
    std::string src_worksFor_file =
        std::string(data_path) + "/src_worksFor.facts";
    vflog::multi_hisa src_worksFor(2, src_worksFor_file.c_str(), global_buffer);
    timer.stop_timer();
    std::cout << "CPU Read time: " << timer.get_spent_time() << std::endl;

    std::cout << "Src Dedup time: " << timer.get_spent_time() << std::endl;

    // st-tgds
    // copy each relation to a relation without src_ prefix
    // create relation first
    timer.start_timer();
    vflog::multi_hisa advisor(2, global_buffer);
    advisor.allocate_newt(src_advisor.get_versioned_size(FULL));
    vflog::column_copy_all(src_advisor, FULL, 0, advisor, NEWT, 0);
    vflog::column_copy_all(src_advisor, FULL, 1, advisor, NEWT, 1);
    advisor.newt_size = src_advisor.get_versioned_size(FULL);
    advisor.total_tuples += src_advisor.get_versioned_size(FULL);
    advisor.newt_self_deduplicate();
    advisor.persist_newt();
    vflog::multi_hisa AssistantProfessor(1, global_buffer);
    AssistantProfessor.allocate_newt(
        src_AssistantProfessor.get_versioned_size(FULL));
    vflog::column_copy_all(src_AssistantProfessor, FULL, 0, AssistantProfessor,
                           NEWT, 0);
    AssistantProfessor.newt_size =
        src_AssistantProfessor.get_versioned_size(FULL);
    AssistantProfessor.total_tuples +=
        src_AssistantProfessor.get_versioned_size(FULL);
    AssistantProfessor.newt_self_deduplicate();
    AssistantProfessor.persist_newt();
    vflog::multi_hisa AssociateProfessor(1, global_buffer);
    AssociateProfessor.allocate_newt(
        src_AssociateProfessor.get_versioned_size(FULL));
    vflog::column_copy_all(src_AssociateProfessor, FULL, 0, AssociateProfessor,
                           NEWT, 0);
    AssociateProfessor.newt_size =
        src_AssociateProfessor.get_versioned_size(FULL);
    AssociateProfessor.total_tuples +=
        src_AssociateProfessor.get_versioned_size(FULL);
    AssociateProfessor.newt_self_deduplicate();
    AssociateProfessor.persist_newt();
    vflog::multi_hisa Course(1, global_buffer);
    Course.allocate_newt(src_Course.get_versioned_size(FULL));
    vflog::column_copy_all(src_Course, FULL, 0, Course, NEWT, 0);
    Course.newt_size = src_Course.get_versioned_size(FULL);
    Course.total_tuples += src_Course.get_versioned_size(FULL);
    Course.newt_self_deduplicate();
    Course.persist_newt();
    vflog::multi_hisa Department(1, global_buffer);
    Department.allocate_newt(src_Department.get_versioned_size(FULL));
    vflog::column_copy_all(src_Department, FULL, 0, Department, NEWT, 0);
    Department.newt_size = src_Department.get_versioned_size(FULL);
    Department.total_tuples += src_Department.get_versioned_size(FULL);
    Department.newt_self_deduplicate();
    Department.persist_newt();
    vflog::multi_hisa doctoralDegreeFrom(2, global_buffer);
    doctoralDegreeFrom.allocate_newt(
        src_doctoralDegreeFrom.get_versioned_size(FULL));
    vflog::column_copy_all(src_doctoralDegreeFrom, FULL, 0, doctoralDegreeFrom,
                           NEWT, 0);
    vflog::column_copy_all(src_doctoralDegreeFrom, FULL, 1, doctoralDegreeFrom,
                           NEWT, 1);
    doctoralDegreeFrom.newt_size =
        src_doctoralDegreeFrom.get_versioned_size(FULL);
    doctoralDegreeFrom.total_tuples +=
        src_doctoralDegreeFrom.get_versioned_size(FULL);
    doctoralDegreeFrom.newt_self_deduplicate();
    doctoralDegreeFrom.persist_newt();
    vflog::multi_hisa emailAddress(2, global_buffer);
    emailAddress.allocate_newt(src_emailAddress.get_versioned_size(FULL));
    vflog::column_copy_all(src_emailAddress, FULL, 0, emailAddress, NEWT, 0);
    vflog::column_copy_all(src_emailAddress, FULL, 1, emailAddress, NEWT, 1);
    emailAddress.newt_size = src_emailAddress.get_versioned_size(FULL);
    emailAddress.total_tuples += src_emailAddress.get_versioned_size(FULL);
    emailAddress.newt_self_deduplicate();
    emailAddress.persist_newt();
    vflog::multi_hisa FullProfessor(1, global_buffer);
    FullProfessor.allocate_newt(src_FullProfessor.get_versioned_size(FULL));
    vflog::column_copy_all(src_FullProfessor, FULL, 0, FullProfessor, NEWT, 0);
    FullProfessor.newt_size = src_FullProfessor.get_versioned_size(FULL);
    FullProfessor.total_tuples += src_FullProfessor.get_versioned_size(FULL);
    FullProfessor.newt_self_deduplicate();
    FullProfessor.persist_newt();
    vflog::multi_hisa GraduateCourse(1, global_buffer);
    GraduateCourse.allocate_newt(src_GraduateCourse.get_versioned_size(FULL));
    vflog::column_copy_all(src_GraduateCourse, FULL, 0, GraduateCourse, NEWT,
                           0);
    GraduateCourse.newt_size = src_GraduateCourse.get_versioned_size(FULL);
    GraduateCourse.total_tuples += src_GraduateCourse.get_versioned_size(FULL);
    GraduateCourse.newt_self_deduplicate();
    GraduateCourse.persist_newt();
    vflog::multi_hisa GraduateStudent(1, global_buffer);
    GraduateStudent.allocate_newt(src_GraduateStudent.get_versioned_size(FULL));
    vflog::column_copy_all(src_GraduateStudent, FULL, 0, GraduateStudent, NEWT,
                           0);
    GraduateStudent.newt_size = src_GraduateStudent.get_versioned_size(FULL);
    GraduateStudent.total_tuples +=
        src_GraduateStudent.get_versioned_size(FULL);
    GraduateStudent.newt_self_deduplicate();
    GraduateStudent.persist_newt();
    vflog::multi_hisa headOf(2, global_buffer);
    headOf.set_index_startegy(1, FULL, vflog::IndexStrategy::EAGER);
    headOf.allocate_newt(src_headOf.get_versioned_size(FULL));
    vflog::column_copy_all(src_headOf, FULL, 0, headOf, NEWT, 0);
    vflog::column_copy_all(src_headOf, FULL, 1, headOf, NEWT, 1);
    headOf.newt_size = src_headOf.get_versioned_size(FULL);
    headOf.total_tuples += src_headOf.get_versioned_size(FULL);
    headOf.newt_self_deduplicate();
    headOf.persist_newt();
    vflog::multi_hisa Lecturer(1, global_buffer);
    Lecturer.allocate_newt(src_Lecturer.get_versioned_size(FULL));
    vflog::column_copy_all(src_Lecturer, FULL, 0, Lecturer, NEWT, 0);
    Lecturer.newt_size = src_Lecturer.get_versioned_size(FULL);
    Lecturer.total_tuples += src_Lecturer.get_versioned_size(FULL);
    Lecturer.newt_self_deduplicate();
    Lecturer.persist_newt();
    vflog::multi_hisa mastersDegreeFrom(2, global_buffer);
    mastersDegreeFrom.allocate_newt(
        src_mastersDegreeFrom.get_versioned_size(FULL));
    vflog::column_copy_all(src_mastersDegreeFrom, FULL, 0, mastersDegreeFrom,
                           NEWT, 0);
    vflog::column_copy_all(src_mastersDegreeFrom, FULL, 1, mastersDegreeFrom,
                           NEWT, 1);
    mastersDegreeFrom.newt_size =
        src_mastersDegreeFrom.get_versioned_size(FULL);
    mastersDegreeFrom.total_tuples +=
        src_mastersDegreeFrom.get_versioned_size(FULL);
    mastersDegreeFrom.newt_self_deduplicate();
    mastersDegreeFrom.persist_newt();
    vflog::multi_hisa memberOf(2, global_buffer);
    memberOf.allocate_newt(src_memberOf.get_versioned_size(FULL));
    vflog::column_copy_all(src_memberOf, FULL, 0, memberOf, NEWT, 0);
    vflog::column_copy_all(src_memberOf, FULL, 1, memberOf, NEWT, 1);
    memberOf.newt_size = src_memberOf.get_versioned_size(NEWT);
    memberOf.total_tuples += src_memberOf.get_versioned_size(NEWT);
    memberOf.newt_self_deduplicate();
    memberOf.persist_newt();
    vflog::multi_hisa name(2, global_buffer);
    name.allocate_newt(src_name.get_versioned_size(FULL));
    vflog::column_copy_all(src_name, FULL, 0, name, NEWT, 0);
    vflog::column_copy_all(src_name, FULL, 1, name, NEWT, 1);
    name.newt_size = src_name.get_versioned_size(FULL);
    name.total_tuples += src_name.get_versioned_size(FULL);
    name.newt_self_deduplicate();
    name.persist_newt();
    vflog::multi_hisa Publication(1, global_buffer);
    Publication.allocate_newt(src_Publication.get_versioned_size(FULL));
    vflog::column_copy_all(src_Publication, FULL, 0, Publication, NEWT, 0);
    Publication.newt_size = src_Publication.get_versioned_size(FULL);
    Publication.total_tuples += src_Publication.get_versioned_size(FULL);
    Publication.newt_self_deduplicate();
    Publication.persist_newt();
    vflog::multi_hisa publicationAuthor(2, global_buffer);
    publicationAuthor.allocate_newt(
        src_publicationAuthor.get_versioned_size(FULL));
    vflog::column_copy_all(src_publicationAuthor, FULL, 0, publicationAuthor,
                           NEWT, 0);
    vflog::column_copy_all(src_publicationAuthor, FULL, 1, publicationAuthor,
                           NEWT, 1);
    publicationAuthor.newt_size =
        src_publicationAuthor.get_versioned_size(FULL);
    publicationAuthor.total_tuples +=
        src_publicationAuthor.get_versioned_size(FULL);
    publicationAuthor.newt_self_deduplicate();
    publicationAuthor.persist_newt();
    vflog::multi_hisa ResearchAssistant(1, global_buffer);
    ResearchAssistant.allocate_newt(
        src_ResearchAssistant.get_versioned_size(FULL));
    vflog::column_copy_all(src_ResearchAssistant, FULL, 0, ResearchAssistant,
                           NEWT, 0);
    ResearchAssistant.newt_size =
        src_ResearchAssistant.get_versioned_size(FULL);
    ResearchAssistant.total_tuples +=
        src_ResearchAssistant.get_versioned_size(FULL);
    ResearchAssistant.newt_self_deduplicate();
    ResearchAssistant.persist_newt();
    vflog::multi_hisa ResearchGroup(1, global_buffer);
    ResearchGroup.allocate_newt(src_ResearchGroup.get_versioned_size(FULL));
    vflog::column_copy_all(src_ResearchGroup, FULL, 0, ResearchGroup, NEWT, 0);
    ResearchGroup.newt_size = src_ResearchGroup.get_versioned_size(FULL);
    ResearchGroup.total_tuples += src_ResearchGroup.get_versioned_size(FULL);
    ResearchGroup.newt_self_deduplicate();
    ResearchGroup.persist_newt();
    vflog::multi_hisa researchInterest(2, global_buffer);
    researchInterest.allocate_newt(
        src_researchInterest.get_versioned_size(FULL));
    vflog::column_copy_all(src_researchInterest, FULL, 0, researchInterest,
                           NEWT, 0);
    vflog::column_copy_all(src_researchInterest, FULL, 1, researchInterest,
                           NEWT, 1);
    researchInterest.newt_size = src_researchInterest.get_versioned_size(FULL);
    researchInterest.total_tuples +=
        src_researchInterest.get_versioned_size(FULL);
    researchInterest.newt_self_deduplicate();
    researchInterest.persist_newt();
    vflog::multi_hisa subOrganizationOf(2, global_buffer);
    subOrganizationOf.allocate_newt(
        src_subOrganizationOf.get_versioned_size(FULL));
    vflog::column_copy_all(src_subOrganizationOf, FULL, 0, subOrganizationOf,
                           NEWT, 0);
    vflog::column_copy_all(src_subOrganizationOf, FULL, 1, subOrganizationOf,
                           NEWT, 1);
    subOrganizationOf.newt_size =
        src_subOrganizationOf.get_versioned_size(FULL);
    subOrganizationOf.total_tuples +=
        src_subOrganizationOf.get_versioned_size(FULL);
    subOrganizationOf.newt_self_deduplicate();
    subOrganizationOf.persist_newt();
    src_subOrganizationOf.persist_newt();
    vflog::multi_hisa takesCourse(2, global_buffer);
    takesCourse.set_index_startegy(1, FULL, vflog::IndexStrategy::EAGER);
    takesCourse.allocate_newt(src_takesCourse.get_versioned_size(FULL));
    vflog::column_copy_all(src_takesCourse, FULL, 0, takesCourse, NEWT, 0);
    vflog::column_copy_all(src_takesCourse, FULL, 1, takesCourse, NEWT, 1);
    takesCourse.newt_size = src_takesCourse.get_versioned_size(FULL);
    takesCourse.total_tuples += src_takesCourse.get_versioned_size(FULL);
    takesCourse.newt_self_deduplicate();
    takesCourse.persist_newt();
    vflog::multi_hisa teacherOf(2, global_buffer);
    teacherOf.allocate_newt(src_teacherOf.get_versioned_size(FULL));
    vflog::column_copy_all(src_teacherOf, FULL, 0, teacherOf, NEWT, 0);
    vflog::column_copy_all(src_teacherOf, FULL, 1, teacherOf, NEWT, 1);
    teacherOf.newt_size = src_teacherOf.get_versioned_size(FULL);
    teacherOf.total_tuples += src_teacherOf.get_versioned_size(FULL);
    teacherOf.newt_self_deduplicate();
    teacherOf.persist_newt();
    vflog::multi_hisa TeachingAssistant(1, global_buffer);
    TeachingAssistant.allocate_newt(
        src_TeachingAssistant.get_versioned_size(FULL));
    vflog::column_copy_all(src_TeachingAssistant, FULL, 0, TeachingAssistant,
                           NEWT, 0);
    TeachingAssistant.newt_size =
        src_TeachingAssistant.get_versioned_size(FULL);
    TeachingAssistant.total_tuples +=
        src_TeachingAssistant.get_versioned_size(FULL);
    TeachingAssistant.newt_self_deduplicate();
    TeachingAssistant.persist_newt();
    vflog::multi_hisa teachingAssistantOf(2, global_buffer);
    teachingAssistantOf.set_index_startegy(1, FULL,
                                           vflog::IndexStrategy::EAGER);
    teachingAssistantOf.allocate_newt(
        src_teachingAssistantOf.get_versioned_size(FULL));
    vflog::column_copy_all(src_teachingAssistantOf, FULL, 0,
                           teachingAssistantOf, NEWT, 0);
    vflog::column_copy_all(src_teachingAssistantOf, FULL, 1,
                           teachingAssistantOf, NEWT, 1);
    teachingAssistantOf.newt_size =
        src_teachingAssistantOf.get_versioned_size(FULL);
    teachingAssistantOf.total_tuples +=
        src_teachingAssistantOf.get_versioned_size(FULL);
    teachingAssistantOf.newt_self_deduplicate();
    teachingAssistantOf.persist_newt();
    vflog::multi_hisa telephone(2, global_buffer);
    telephone.allocate_newt(src_telephone.get_versioned_size(FULL));
    vflog::column_copy_all(src_telephone, FULL, 0, telephone, NEWT, 0);
    vflog::column_copy_all(src_telephone, FULL, 1, telephone, NEWT, 1);
    telephone.newt_size = src_telephone.get_versioned_size(FULL);
    telephone.total_tuples += src_telephone.get_versioned_size(FULL);
    telephone.newt_self_deduplicate();
    telephone.persist_newt();
    vflog::multi_hisa undergraduateDegreeFrom(2, global_buffer);
    undergraduateDegreeFrom.allocate_newt(
        src_undergraduateDegreeFrom.get_versioned_size(FULL));
    vflog::column_copy_all(src_undergraduateDegreeFrom, FULL, 0,
                           undergraduateDegreeFrom, NEWT, 0);
    vflog::column_copy_all(src_undergraduateDegreeFrom, FULL, 1,
                           undergraduateDegreeFrom, NEWT, 1);
    undergraduateDegreeFrom.newt_size =
        src_undergraduateDegreeFrom.get_versioned_size(FULL);
    undergraduateDegreeFrom.total_tuples +=
        src_undergraduateDegreeFrom.get_versioned_size(FULL);
    undergraduateDegreeFrom.newt_self_deduplicate();
    undergraduateDegreeFrom.persist_newt();
    vflog::multi_hisa UndergraduateStudent(1, global_buffer);
    UndergraduateStudent.allocate_newt(
        src_UndergraduateStudent.get_versioned_size(FULL));
    vflog::column_copy_all(src_UndergraduateStudent, FULL, 0,
                           UndergraduateStudent, NEWT, 0);
    UndergraduateStudent.newt_size =
        src_UndergraduateStudent.get_versioned_size(FULL);
    UndergraduateStudent.total_tuples +=
        src_UndergraduateStudent.get_versioned_size(FULL);
    UndergraduateStudent.newt_self_deduplicate();
    UndergraduateStudent.persist_newt();
    vflog::multi_hisa University(1, global_buffer);
    University.allocate_newt(src_University.get_versioned_size(FULL));
    vflog::column_copy_all(src_University, FULL, 0, University, NEWT, 0);
    University.newt_size = src_University.get_versioned_size(FULL);
    University.total_tuples += src_University.get_versioned_size(FULL);
    University.newt_self_deduplicate();
    University.persist_newt();
    vflog::multi_hisa worksFor(2, global_buffer);
    worksFor.set_index_startegy(1, FULL, vflog::IndexStrategy::EAGER);
    // worksFor.set_index_startegy(1, DELTA, vflog::IndexStrategy::EAGER);
    worksFor.allocate_newt(src_worksFor.get_versioned_size(FULL));
    vflog::column_copy_all(src_worksFor, FULL, 0, worksFor, NEWT, 0);
    vflog::column_copy_all(src_worksFor, FULL, 1, worksFor, NEWT, 1);
    worksFor.newt_size = src_worksFor.get_versioned_size(FULL);
    worksFor.total_tuples += src_worksFor.get_versioned_size(FULL);
    worksFor.newt_self_deduplicate();
    worksFor.persist_newt();
    timer.stop_timer();

    // copy time
    std::cout << "Copy time: " << timer.get_spent_time() << std::endl;

    timer.start_timer();
    // t-tgds
    vflog::multi_hisa Person(1, global_buffer);
    Person.set_index_startegy(0, DELTA, vflog::IndexStrategy::EAGER);
    // Person(?X) :- advisor(?X, ?X1) .
    Person.allocate_newt(advisor.get_versioned_size(FULL));
    vflog::column_copy_all(advisor, FULL, 0, Person, NEWT, 0);
    Person.newt_size += advisor.get_versioned_size(FULL);
    Person.total_tuples += advisor.get_versioned_size(FULL);
    // Professor(?X1) :- advisor(?X, ?X1) .
    vflog::multi_hisa Professor(1, global_buffer);
    Professor.allocate_newt(advisor.get_versioned_size(FULL));
    vflog::column_copy_all(advisor, FULL, 1, Professor, NEWT, 0);
    Professor.newt_size += advisor.get_versioned_size(FULL);
    Professor.total_tuples += advisor.get_versioned_size(FULL);
    // Professor(?X) :- AssistantProfessor(?X) .
    Professor.allocate_newt(AssistantProfessor.get_versioned_size(FULL));
    vflog::column_copy_all(AssistantProfessor, FULL, 0, Professor, NEWT, 0,
                           true);
    Professor.newt_size += AssistantProfessor.get_versioned_size(FULL);
    Professor.total_tuples += AssistantProfessor.get_versioned_size(FULL);
    // Person(?X) :- AssociateProfessor(?X) .
    Person.allocate_newt(AssociateProfessor.get_versioned_size(FULL));
    vflog::column_copy_all(AssociateProfessor, FULL, 0, Person, NEWT, 0, true);
    Person.newt_size += AssociateProfessor.get_versioned_size(FULL);
    Person.total_tuples += AssociateProfessor.get_versioned_size(FULL);
    Person.newt_self_deduplicate();
    // Organization(?X) :- Department(?X) .
    vflog::multi_hisa Organization(1, global_buffer);
    Organization.allocate_newt(Department.get_versioned_size(FULL));
    vflog::column_copy_all(Department, FULL, 0, Organization, NEWT, 0);
    Organization.newt_size += Department.get_versioned_size(FULL);
    Organization.total_tuples += Department.get_versioned_size(FULL);
    // Person(?X) :- doctoralDegreeFrom(?X, ?X1) .
    Person.allocate_newt(doctoralDegreeFrom.get_versioned_size(FULL));
    vflog::column_copy_all(doctoralDegreeFrom, FULL, 0, Person, NEWT, 0, true);
    Person.newt_size += doctoralDegreeFrom.get_versioned_size(FULL);
    Person.total_tuples += doctoralDegreeFrom.get_versioned_size(FULL);
    // University(?X1) :- doctoralDegreeFrom(?X, ?X1) .
    University.allocate_newt(doctoralDegreeFrom.get_versioned_size(FULL));
    vflog::column_copy_all(doctoralDegreeFrom, FULL, 1, University, NEWT, 0);
    University.newt_size += doctoralDegreeFrom.get_versioned_size(FULL);
    University.total_tuples += doctoralDegreeFrom.get_versioned_size(FULL);
    // degreeFrom(?X, ?Y) :- doctoralDegreeFrom(?X, ?Y) .
    vflog::multi_hisa degreeFrom(2, global_buffer);
    degreeFrom.allocate_newt(doctoralDegreeFrom.get_versioned_size(FULL));
    vflog::column_copy_all(doctoralDegreeFrom, FULL, 0, degreeFrom, NEWT, 0);
    vflog::column_copy_all(doctoralDegreeFrom, FULL, 1, degreeFrom, NEWT, 1);
    degreeFrom.newt_size += doctoralDegreeFrom.get_versioned_size(FULL);
    degreeFrom.total_tuples += doctoralDegreeFrom.get_versioned_size(FULL);
    // Person(?X) :- emailAddress(?X, ?X1) .
    Person.allocate_newt(emailAddress.get_versioned_size(FULL));
    vflog::column_copy_all(emailAddress, FULL, 0, Person, NEWT, 0);
    Person.newt_size += emailAddress.get_versioned_size(FULL);
    Person.total_tuples += emailAddress.get_versioned_size(FULL);
    Person.newt_self_deduplicate();
    // Professor(?X) :- FullProfessor(?X) .
    Professor.allocate_newt(FullProfessor.get_versioned_size(FULL));
    vflog::column_copy_all(FullProfessor, FULL, 0, Professor, NEWT, 0, true);
    Professor.newt_size += FullProfessor.get_versioned_size(FULL);
    Professor.total_tuples += FullProfessor.get_versioned_size(FULL);
    // Course(?X) :- GraduateCourse(?X) .
    Course.allocate_newt(GraduateCourse.get_versioned_size(FULL));
    vflog::column_copy_all(GraduateCourse, FULL, 0, Course, NEWT, 0);
    Course.newt_size += GraduateCourse.get_versioned_size(FULL);
    Course.total_tuples += GraduateCourse.get_versioned_size(FULL);
    // Person(?X) :- GraduateStudent(?X) .
    Person.allocate_newt(GraduateStudent.get_versioned_size(FULL));
    vflog::column_copy_all(GraduateStudent, FULL, 0, Person, NEWT, 0);
    Person.newt_size += GraduateStudent.get_versioned_size(FULL);
    Person.total_tuples += GraduateStudent.get_versioned_size(FULL);
    Person.newt_self_deduplicate();
    // takesCourse(?X, !Y), GraduateCourse(!Y) :- GraduateStudent(?X) .

    // Faculty(?X) :- Lecturer(?X) .
    vflog::multi_hisa Faculty(1, global_buffer);
    Faculty.allocate_newt(Lecturer.get_versioned_size(FULL));
    vflog::column_copy_all(Lecturer, FULL, 0, Faculty, NEWT, 0);
    Faculty.newt_size += Lecturer.get_versioned_size(FULL);
    Faculty.total_tuples += Lecturer.get_versioned_size(FULL);
    // Person(?X) :- mastersDegreeFrom(?X, ?X1) .
    Person.allocate_newt(mastersDegreeFrom.get_versioned_size(FULL));
    vflog::column_copy_all(mastersDegreeFrom, FULL, 0, Person, NEWT, 0, true);
    Person.newt_size += mastersDegreeFrom.get_versioned_size(FULL);
    Person.total_tuples += mastersDegreeFrom.get_versioned_size(FULL);
    Person.newt_self_deduplicate();
    // University(?X1) :- mastersDegreeFrom(?X, ?X1) .
    University.allocate_newt(mastersDegreeFrom.get_versioned_size(FULL));
    vflog::column_copy_all(mastersDegreeFrom, FULL, 1, University, NEWT, 0);
    University.newt_size += mastersDegreeFrom.get_versioned_size(FULL);
    University.total_tuples += mastersDegreeFrom.get_versioned_size(FULL);
    // degreeFrom(?X, ?Y) :- mastersDegreeFrom(?X, ?Y) .
    degreeFrom.allocate_newt(mastersDegreeFrom.get_versioned_size(FULL));
    vflog::column_copy_all(mastersDegreeFrom, FULL, 0, degreeFrom, NEWT, 0);
    vflog::column_copy_all(mastersDegreeFrom, FULL, 1, degreeFrom, NEWT, 1);
    degreeFrom.newt_size += mastersDegreeFrom.get_versioned_size(FULL);
    degreeFrom.total_tuples += mastersDegreeFrom.get_versioned_size(FULL);
    // member(?X, ?Y) :- memberOf(?Y, ?X) .
    vflog::multi_hisa member(2, global_buffer);
    member.allocate_newt(memberOf.get_versioned_size(FULL));
    vflog::column_copy_all(memberOf, FULL, 1, member, NEWT, 0);
    vflog::column_copy_all(memberOf, FULL, 0, member, NEWT, 1);
    member.newt_size += memberOf.get_versioned_size(FULL);
    member.total_tuples += memberOf.get_versioned_size(FULL);
    // Person(?X1) :- publicationAuthor(?X, ?X1) .
    Person.allocate_newt(publicationAuthor.get_versioned_size(FULL));
    vflog::column_copy_all(publicationAuthor, FULL, 1, Person, NEWT, 0);
    Person.newt_size += publicationAuthor.get_versioned_size(FULL);
    Person.total_tuples += publicationAuthor.get_versioned_size(FULL);
    Person.newt_self_deduplicate();
    // Publication(?X) :- publicationAuthor(?X, ?X1) .
    Publication.allocate_newt(publicationAuthor.get_versioned_size(FULL));
    vflog::column_copy_all(publicationAuthor, FULL, 0, Publication, NEWT, 0);
    Publication.newt_size += publicationAuthor.get_versioned_size(FULL);
    Publication.total_tuples += publicationAuthor.get_versioned_size(FULL);
    // Person(?X) :- ResearchAssistant(?X) .
    Person.allocate_newt(ResearchAssistant.get_versioned_size(FULL));
    vflog::column_copy_all(ResearchAssistant, FULL, 0, Person, NEWT, 0);
    Person.newt_size += ResearchAssistant.get_versioned_size(FULL);
    Person.total_tuples += ResearchAssistant.get_versioned_size(FULL);
    Person.newt_self_deduplicate();

    // Course(?X1) :- teacherOf(?X, ?X1) .
    Course.allocate_newt(teacherOf.get_versioned_size(FULL));
    vflog::column_copy_all(teacherOf, FULL, 1, Course, NEWT, 0);
    Course.newt_size += teacherOf.get_versioned_size(FULL);
    Course.total_tuples += teacherOf.get_versioned_size(FULL);
    // Faculty(?X) :- teacherOf(?X, ?X1) .
    Faculty.allocate_newt(teacherOf.get_versioned_size(FULL));
    vflog::column_copy_all(teacherOf, FULL, 0, Faculty, NEWT, 0);
    Faculty.newt_size += teacherOf.get_versioned_size(FULL);
    Faculty.total_tuples += teacherOf.get_versioned_size(FULL);

    // Person(?X) :- telephone(?X, ?X1) .
    Person.allocate_newt(telephone.get_versioned_size(FULL));
    vflog::column_copy_all(telephone, FULL, 0, Person, NEWT, 0);
    Person.newt_size += telephone.get_versioned_size(FULL);
    Person.total_tuples += telephone.get_versioned_size(FULL);
    Person.newt_self_deduplicate();
    // Person(?X) :- undergraduateDegreeFrom(?X, ?X1) .
    Person.allocate_newt(undergraduateDegreeFrom.get_versioned_size(FULL));
    vflog::column_copy_all(undergraduateDegreeFrom, FULL, 0, Person, NEWT, 0,
                           true);
    Person.newt_size += undergraduateDegreeFrom.get_versioned_size(FULL);
    Person.total_tuples += undergraduateDegreeFrom.get_versioned_size(FULL);
    // University(?X1) :- undergraduateDegreeFrom(?X, ?X1) .
    University.allocate_newt(undergraduateDegreeFrom.get_versioned_size(FULL));
    vflog::column_copy_all(undergraduateDegreeFrom, FULL, 1, University, NEWT,
                           0);
    University.newt_size += undergraduateDegreeFrom.get_versioned_size(FULL);
    University.total_tuples += undergraduateDegreeFrom.get_versioned_size(FULL);
    // degreeFrom(?X, ?Y) :- undergraduateDegreeFrom(?X, ?Y) .
    degreeFrom.allocate_newt(undergraduateDegreeFrom.get_versioned_size(FULL));
    vflog::column_copy_all(undergraduateDegreeFrom, FULL, 0, degreeFrom, NEWT,
                           0);
    vflog::column_copy_all(undergraduateDegreeFrom, FULL, 1, degreeFrom, NEWT,
                           1);
    degreeFrom.newt_size += undergraduateDegreeFrom.get_versioned_size(FULL);
    degreeFrom.total_tuples += undergraduateDegreeFrom.get_versioned_size(FULL);
    // Student(?X) :- UndergraduateStudent(?X) .
    vflog::multi_hisa Student(1, global_buffer);
    Student.allocate_newt(UndergraduateStudent.get_versioned_size(FULL));
    vflog::column_copy_all(UndergraduateStudent, FULL, 0, Student, NEWT, 0);
    Student.newt_size += UndergraduateStudent.get_versioned_size(FULL);
    Student.total_tuples += UndergraduateStudent.get_versioned_size(FULL);

    // Person(?X) :- TeachingAssistant(?X) .
    Person.allocate_newt(TeachingAssistant.get_versioned_size(FULL));
    vflog::column_copy_all(TeachingAssistant, FULL, 0, Person, NEWT, 0);
    Person.newt_size += TeachingAssistant.get_versioned_size(FULL);
    Person.total_tuples += TeachingAssistant.get_versioned_size(FULL);

    // deduplicate all non-recurisve relations
    Person.newt_self_deduplicate();
    Professor.newt_self_deduplicate();
    Organization.newt_self_deduplicate();
    University.newt_self_deduplicate();
    degreeFrom.newt_self_deduplicate();
    Publication.newt_self_deduplicate();
    member.newt_self_deduplicate();
    Faculty.newt_self_deduplicate();
    Student.newt_self_deduplicate();

    Person.persist_newt();
    Professor.persist_newt();
    Organization.persist_newt();
    University.persist_newt();
    degreeFrom.persist_newt();
    Publication.persist_newt();
    member.persist_newt();
    Faculty.persist_newt();
    Student.persist_newt();

    vflog::multi_hisa Chair(1, global_buffer);
    vflog::multi_hisa hasAlumnus(2, global_buffer);
    vflog::multi_hisa Employee(1, global_buffer);
    Chair.uid = 0;
    Employee.uid = 1;
    Student.uid = 2;
    ResearchAssistant.uid = 3;
    TeachingAssistant.uid = 4;

    // prepare join buffer
    auto matched_x_ptr = std::make_shared<vflog::device_indices_t>();
    auto matched_x1_ptr = std::make_shared<vflog::device_indices_t>();
    auto matched_y_ptr = std::make_shared<vflog::device_indices_t>();
    vflog::host_buf_ref_t cached;
    size_t iteration = 0;
    while (true) {
        cached.clear();
        std::cout << "Iteration " << iteration << std::endl;

        // headOf(?X, !Y), Department(!Y) :- Chair(?X) .
        // Person(?X) :- Chair(?X) .
        Person.allocate_newt(Chair.get_versioned_size(DELTA));
        vflog::column_copy_all(Chair, DELTA, 0, Person, NEWT, 0, true);
        Person.newt_size += Chair.get_versioned_size(DELTA);
        Person.total_tuples += Chair.get_versioned_size(DELTA);
        std::cout << "Person(?X) :- Chair(?X) ." << std::endl;
        // Professor(?X) :- Chair(?X) .
        Professor.allocate_newt(Chair.get_versioned_size(DELTA));
        vflog::column_copy_all(Chair, DELTA, 0, Professor, NEWT, 0, true);
        Professor.newt_size += Chair.get_versioned_size(DELTA);
        Professor.total_tuples += Chair.get_versioned_size(DELTA);
        std::cout << "Professor(?X) :- Chair(?X) ." << std::endl;

        // Person(?X) :- degreeFrom(?X, ?X1) .
        Person.allocate_newt(degreeFrom.get_versioned_size(DELTA));
        vflog::column_copy_all(degreeFrom, DELTA, 0, Person, NEWT, 0, true);
        Person.newt_size += degreeFrom.get_versioned_size(DELTA);
        Person.total_tuples += degreeFrom.get_versioned_size(DELTA);
        Person.newt_self_deduplicate();
        std::cout << "Person(?X) :- degreeFrom(?X, ?X1) ." << std::endl;
        // University(?X1) :- degreeFrom(?X, ?X1) .
        University.allocate_newt(degreeFrom.get_versioned_size(DELTA));
        vflog::column_copy_all(degreeFrom, DELTA, 1, University, NEWT, 0, true);
        University.newt_size += degreeFrom.get_versioned_size(DELTA);
        University.total_tuples += degreeFrom.get_versioned_size(DELTA);
        // hasAlumnus(?X, ?Y) :- degreeFrom(?Y, ?X) .
        hasAlumnus.allocate_newt(degreeFrom.get_versioned_size(DELTA));
        vflog::column_copy_all(degreeFrom, DELTA, 0, hasAlumnus, NEWT, 1, true);
        vflog::column_copy_all(degreeFrom, DELTA, 1, hasAlumnus, NEWT, 0, true);
        hasAlumnus.newt_size += degreeFrom.get_versioned_size(DELTA);
        hasAlumnus.total_tuples += degreeFrom.get_versioned_size(DELTA);

        // Person(?X) :- Employee(?X) .
        Person.allocate_newt(Employee.get_versioned_size(DELTA));
        vflog::column_copy_all(Employee, DELTA, 0, Person, NEWT, 0, true);
        Person.newt_size += Employee.get_versioned_size(DELTA);
        Person.total_tuples += Employee.get_versioned_size(DELTA);

        // worksFor(?X, !Y), Organization(!Y) :- Employee(?X) .

        // Employee(?X) :- Faculty(?X) .
        Employee.allocate_newt(Faculty.get_versioned_size(DELTA));
        vflog::column_copy_all(Faculty, DELTA, 0, Employee, NEWT, 0, true);
        Employee.newt_size += Faculty.get_versioned_size(DELTA);
        Employee.total_tuples += Faculty.get_versioned_size(DELTA);

        // Person(?X1) :- hasAlumnus(?X, ?X1) .
        Person.allocate_newt(hasAlumnus.get_versioned_size(DELTA));
        vflog::column_copy_all(hasAlumnus, DELTA, 1, Person, NEWT, 0, true);
        Person.newt_size += hasAlumnus.get_versioned_size(DELTA);
        Person.total_tuples += hasAlumnus.get_versioned_size(DELTA);
        Person.newt_self_deduplicate();
        // University(?X) :- hasAlumnus(?X, ?X1) .
        University.allocate_newt(hasAlumnus.get_versioned_size(DELTA));
        vflog::column_copy_all(hasAlumnus, DELTA, 0, University, NEWT, 0, true);
        University.newt_size += hasAlumnus.get_versioned_size(DELTA);
        University.total_tuples += hasAlumnus.get_versioned_size(DELTA);
        // degreeFrom(?X, ?Y) :- hasAlumnus(?Y, ?X) .
        degreeFrom.allocate_newt(hasAlumnus.get_versioned_size(DELTA));
        vflog::column_copy_all(hasAlumnus, DELTA, 1, degreeFrom, NEWT, 0, true);
        vflog::column_copy_all(hasAlumnus, DELTA, 0, degreeFrom, NEWT, 1, true);
        degreeFrom.newt_size += hasAlumnus.get_versioned_size(DELTA);
        degreeFrom.total_tuples += hasAlumnus.get_versioned_size(DELTA);

        // worksFor(?X, ?Y) :- headOf(?X, ?Y) .
        worksFor.allocate_newt(headOf.get_versioned_size(DELTA));
        vflog::column_copy_all(headOf, DELTA, 0, worksFor, NEWT, 0, true);
        vflog::column_copy_all(headOf, DELTA, 1, worksFor, NEWT, 1, true);
        worksFor.newt_size += headOf.get_versioned_size(DELTA);
        worksFor.total_tuples += headOf.get_versioned_size(DELTA);
        // Organization(?X) :- member(?X, ?X1) .
        Organization.allocate_newt(member.get_versioned_size(DELTA));
        vflog::column_copy_all(member, DELTA, 1, Organization, NEWT, 0, true);
        Organization.newt_size += member.get_versioned_size(DELTA);
        Organization.total_tuples += member.get_versioned_size(DELTA);
        // Person(?X1) :- member(?X, ?X1) .
        Person.allocate_newt(member.get_versioned_size(DELTA));
        vflog::column_copy_all(member, DELTA, 1, Person, NEWT, 0, true);
        Person.newt_size += member.get_versioned_size(DELTA);
        Person.total_tuples += member.get_versioned_size(DELTA);
        Person.newt_self_deduplicate();
        // memberOf(?X, ?Y) :- member(?Y, ?X) .
        memberOf.allocate_newt(member.get_versioned_size(DELTA));
        vflog::column_copy_all(member, DELTA, 0, memberOf, NEWT, 1, true);
        vflog::column_copy_all(member, DELTA, 1, memberOf, NEWT, 0, true);
        memberOf.newt_size += member.get_versioned_size(DELTA);
        memberOf.total_tuples += member.get_versioned_size(DELTA);

        // Chair(?X) :- Person(?X), headOf(?X, ?X1), Department(?X1) .
        // TODO:
        cached["Person"] = matched_x_ptr;
        cached["Person"]->resize(Person.get_versioned_size(DELTA));
        thrust::sequence(cached["Person"]->begin(), cached["Person"]->end());
        vflog::column_join(headOf, FULL, 0, Person, DELTA, 0, cached, "Person",
                           matched_x1_ptr);
        cached["headOf"] = matched_x1_ptr;
        vflog::column_join(Department, FULL, 0, headOf, FULL, 1, cached,
                           "headOf", matched_y_ptr, true);
        Chair.allocate_newt(cached["Person"]->size());
        vflog::column_copy(Person, DELTA, 0, Chair, NEWT, 0, cached["Person"]);
        Chair.newt_size += cached["Person"]->size();
        Chair.total_tuples += cached["Person"]->size();
        cached.clear();
        std::cout
            << "Chair(?X) :- Person(?X), headOf(?X, ?X1), Department(?X1) ."
            << std::endl;

        // Student(?X) :- Person(?X), takesCourse(?X, ?X1), Course(?X1) .
        // TODO:
        cached["Person"] = matched_x_ptr;
        cached["Person"]->resize(Person.get_versioned_size(DELTA));
        thrust::sequence(cached["Person"]->begin(), cached["Person"]->end());
        vflog::column_join(takesCourse, FULL, 0, Person, DELTA, 0, cached,
                           "Person", matched_x1_ptr);
        cached["takesCourse"] = matched_x1_ptr;
        vflog::column_join(Course, FULL, 0, takesCourse, FULL, 1, cached,
                           "takesCourse", matched_y_ptr, true);
        Student.allocate_newt(cached["Person"]->size());
        vflog::column_copy(Person, DELTA, 0, Student, NEWT, 0,
                           cached["Person"]);
        Student.newt_size += cached["Person"]->size();
        Student.total_tuples += cached["Person"]->size();
        cached.clear();
        std::cout << "Student(?X) :- Person(?X), takesCourse(?X, ?X1), "
                     "Course(?X1) ."
                  << std::endl;

        // TeachingAssistant(?X) :- Person(?X), teachingAssistantOf(?X, ?X1),
        // Course(?X1) .
        // TODO:
        cached["Person"] = matched_x_ptr;
        cached["Person"]->resize(Person.get_versioned_size(DELTA));
        thrust::sequence(cached["Person"]->begin(), cached["Person"]->end());
        vflog::column_join(teachingAssistantOf, FULL, 0, Person, DELTA, 0,
                           cached, "Person", matched_x1_ptr);
        cached["teachingAssistantOf"] = matched_x1_ptr;
        vflog::column_join(Course, FULL, 0, teachingAssistantOf, FULL, 1,
                           cached, "teachingAssistantOf", matched_y_ptr, true);
        TeachingAssistant.allocate_newt(cached["Person"]->size());
        vflog::column_copy(Person, DELTA, 0, TeachingAssistant, NEWT, 0,
                           cached["Person"]);
        TeachingAssistant.newt_size += cached["Person"]->size();
        TeachingAssistant.total_tuples += cached["Person"]->size();
        cached.clear();
        std::cout << "TeachingAssistant(?X) :- Person(?X), "
                     "teachingAssistantOf(?X, ?X1), Course(?X1) ."
                  << std::endl;

        // Employee(?X) :- Person(?X), worksFor(?X, ?X1),
        // Organization(?X1) .
        // TODO:
        cached["Person"] = matched_x_ptr;
        cached["Person"]->resize(Person.get_versioned_size(DELTA));
        thrust::sequence(cached["Person"]->begin(), cached["Person"]->end());
        vflog::column_join(worksFor, FULL, 0, Person, DELTA, 0, cached,
                           "Person", matched_x1_ptr);
        cached["worksFor"] = matched_x1_ptr;
        vflog::column_join(Organization, FULL, 0, worksFor, FULL, 1, cached,
                           "worksFor", matched_y_ptr, true);
        Employee.allocate_newt(cached["Person"]->size());
        vflog::column_copy(Person, DELTA, 0, Employee, NEWT, 0,
                           cached["Person"]);
        Employee.newt_size += cached["Person"]->size();
        Employee.total_tuples += cached["Person"]->size();
        cached.clear();
        std::cout << "Employee(?X) :- Person(?X), worksFor(?X, ?X1), "
                     "Organization(?X1) . ver 1"
                  << std::endl;
        cached["worksFor"] = matched_x_ptr;
        cached["worksFor"]->resize(worksFor.get_versioned_size(DELTA));
        thrust::sequence(cached["worksFor"]->begin(),
                         cached["worksFor"]->end());
        vflog::column_join(Person, FULL, 0, worksFor, DELTA, 0, cached,
                           "worksFor", matched_x1_ptr);
        cached["Person"] = matched_x1_ptr;
        vflog::column_join(Organization, FULL, 0, worksFor, DELTA, 1, cached,
                           "worksFor", matched_y_ptr, true);
        Employee.allocate_newt(cached["Person"]->size());
        vflog::column_copy(Person, FULL, 0, Employee, NEWT, 0,
                           cached["Person"]);
        Employee.newt_size += cached["Person"]->size();
        Employee.total_tuples += cached["Person"]->size();
        cached.clear();
        std::cout << "Employee(?X) :- Person(?X), worksFor(?X, ?X1), "
                     "Organization(?X1) . ver 2"
                  << std::endl;

        // Faculty(?X) :- Professor(?X) .
        Faculty.allocate_newt(Professor.get_versioned_size(DELTA));
        vflog::column_copy_all(Professor, DELTA, 0, Faculty, NEWT, 0, true);
        Faculty.newt_size += Professor.get_versioned_size(DELTA);
        Faculty.total_tuples += Professor.get_versioned_size(DELTA);
        std::cout << "Faculty(?X) :- Professor(?X) ." << std::endl;

        // Person(?X) :- Student(?X) .
        Person.allocate_newt(Student.get_versioned_size(DELTA));
        vflog::column_copy_all(Student, DELTA, 0, Person, NEWT, 0, true);
        Person.newt_size += Student.get_versioned_size(DELTA);
        Person.total_tuples += Student.get_versioned_size(DELTA);
        Person.newt_self_deduplicate();
        std::cout << "Person(?X) :- Student(?X) ." << std::endl;

        Organization.allocate_newt(subOrganizationOf.get_versioned_size(DELTA));
        vflog::column_copy_all(subOrganizationOf, DELTA, 0, Organization, NEWT,
                               0, true);
        Organization.newt_size += subOrganizationOf.get_versioned_size(DELTA);
        Organization.total_tuples +=
            subOrganizationOf.get_versioned_size(DELTA);
        std::cout << "Organization(?X) :- subOrganizationOf(?X, ?X1) ."
                  << std::endl;
        // Organization(?X1) :- subOrganizationOf(?X, ?X1) .
        Organization.allocate_newt(subOrganizationOf.get_versioned_size(DELTA));
        vflog::column_copy_all(subOrganizationOf, DELTA, 1, Organization, NEWT,
                               0, true);
        Organization.newt_size += subOrganizationOf.get_versioned_size(DELTA);
        Organization.total_tuples +=
            subOrganizationOf.get_versioned_size(DELTA);
        std::cout << "Organization(?X1) :- subOrganizationOf(?X, ?X1) ."
                  << std::endl;
        // subOrganizationOf(?X, ?Z) :- src_subOrganizationOf(?X, ?Y),
        // subOrganizationOf(?Y, ?Z) .
        // TODO:
        cached["subOrganizationOf"] = matched_x_ptr;
        cached["subOrganizationOf"]->resize(
            subOrganizationOf.get_versioned_size(DELTA));
        thrust::sequence(cached["subOrganizationOf"]->begin(),
                         cached["subOrganizationOf"]->end());
        vflog::column_join(src_subOrganizationOf, FULL, 1, subOrganizationOf,
                           DELTA, 0, cached, "subOrganizationOf",
                           matched_x1_ptr);
        cached["src_subOrganizationOf"] = matched_x1_ptr;
        subOrganizationOf.allocate_newt(
            cached["src_subOrganizationOf"]->size());
        vflog::column_copy(src_subOrganizationOf, FULL, 0, subOrganizationOf,
                           NEWT, 0, cached["src_subOrganizationOf"]);
        vflog::column_copy(subOrganizationOf, DELTA, 1, subOrganizationOf, NEWT,
                           1, cached["subOrganizationOf"]);
        subOrganizationOf.newt_size += cached["subOrganizationOf"]->size();
        subOrganizationOf.total_tuples += cached["subOrganizationOf"]->size();
        cached.clear();
        std::cout << "subOrganizationOf(?X, ?Z) :- src_subOrganizationOf(?X, "
                     "?Y), subOrganizationOf(?Y, ?Z) ."
                  << std::endl;

        // Course(?X1) :- teachingAssistantOf(?X, ?X1) .
        Course.allocate_newt(teachingAssistantOf.get_versioned_size(DELTA));
        vflog::column_copy_all(teachingAssistantOf, DELTA, 1, Course, NEWT, 0,
                               true);
        Course.newt_size += teachingAssistantOf.get_versioned_size(DELTA);
        Course.total_tuples += teachingAssistantOf.get_versioned_size(DELTA);

        // Organization(?X) :- University(?X) .
        Organization.allocate_newt(University.get_versioned_size(DELTA));
        vflog::column_copy_all(University, DELTA, 0, Organization, NEWT, 0,
                               true);
        Organization.newt_size += University.get_versioned_size(DELTA);
        Organization.total_tuples += University.get_versioned_size(DELTA);

        // deduplicate all recurisve relations and persist
        Chair.newt_self_deduplicate();
        hasAlumnus.newt_self_deduplicate();
        Employee.newt_self_deduplicate();
        worksFor.newt_self_deduplicate();
        subOrganizationOf.newt_self_deduplicate();
        teachingAssistantOf.newt_self_deduplicate();
        Organization.newt_self_deduplicate();
        memberOf.newt_self_deduplicate();
        Person.newt_self_deduplicate();
        Professor.newt_self_deduplicate();
        University.newt_self_deduplicate();
        degreeFrom.newt_self_deduplicate();
        Faculty.newt_self_deduplicate();

        std::cout << "Iteration " << iteration << " deduplication done."
                  << std::endl;
        // persist all recursive relations
        Chair.persist_newt();
        hasAlumnus.persist_newt();
        Employee.persist_newt();
        worksFor.persist_newt();
        subOrganizationOf.persist_newt();
        teachingAssistantOf.persist_newt();
        Organization.persist_newt();
        memberOf.persist_newt();
        Person.persist_newt();
        Professor.persist_newt();
        University.persist_newt();
        degreeFrom.persist_newt();
        Faculty.persist_newt();

        size_t total_delta_size =
            Chair.get_versioned_size(DELTA) +
            hasAlumnus.get_versioned_size(DELTA) +
            Employee.get_versioned_size(DELTA) +
            worksFor.get_versioned_size(DELTA) +
            subOrganizationOf.get_versioned_size(DELTA) +
            teachingAssistantOf.get_versioned_size(DELTA) +
            Organization.get_versioned_size(DELTA) +
            memberOf.get_versioned_size(DELTA) +
            Person.get_versioned_size(DELTA) +
            Professor.get_versioned_size(DELTA) +
            University.get_versioned_size(DELTA) +
            degreeFrom.get_versioned_size(DELTA) +
            Faculty.get_versioned_size(DELTA);

        if (total_delta_size == 0) {
            break;
        }

        iteration++;
    }
    cached.clear();

    // worksFor(?X, !Y), ResearchGroup(!Y) :- ResearchAssistant(?X) .
    // This is same as
    //  ResearchGroup(@autoince), WorksFor(?X, @autoince) :-
    //     !WorksFor(?X, _),  ResearchAssistant(?X) .
    cached["ResearchAssistant"] = matched_x_ptr;
    cached["ResearchAssistant"]->resize(
        ResearchAssistant.get_versioned_size(FULL));
    thrust::sequence(cached["ResearchAssistant"]->begin(),
                     cached["ResearchAssistant"]->end());
    vflog::column_negate(worksFor, FULL, 0, ResearchAssistant, FULL, 0, cached,
                         "ResearchAssistant");
    ResearchGroup.allocate_newt(cached["ResearchAssistant"]->size());
    vflog::column_copy_indices(ResearchAssistant, FULL, 0, ResearchGroup, NEWT,
                               0, cached["ResearchAssistant"]);
    ResearchGroup.newt_size += cached["ResearchAssistant"]->size();
    ResearchGroup.total_tuples += cached["ResearchAssistant"]->size();
    worksFor.allocate_newt(cached["ResearchAssistant"]->size());
    vflog::column_copy(ResearchAssistant, FULL, 0, worksFor, NEWT, 0,
                       cached["ResearchAssistant"]);
    vflog::column_copy_indices(ResearchAssistant, FULL, 0, worksFor, NEWT, 1,
                               cached["ResearchAssistant"]);
    worksFor.newt_size += cached["ResearchAssistant"]->size();
    worksFor.total_tuples += cached["ResearchAssistant"]->size();
    cached.clear();
    ResearchGroup.newt_self_deduplicate();
    worksFor.newt_self_deduplicate();
    ResearchGroup.persist_newt();
    worksFor.persist_newt();
    std::cout
        << "worksFor(?X, !Y), ResearchGroup(!Y) :- ResearchAssistant(?X) ."
        << std::endl;

    // memberOf(?X, ?Y) :- worksFor(?X, ?Y) .
    memberOf.allocate_newt(worksFor.get_versioned_size(FULL));
    vflog::column_copy_all(worksFor, FULL, 0, memberOf, NEWT, 0, true);
    vflog::column_copy_all(worksFor, FULL, 1, memberOf, NEWT, 1, true);
    memberOf.newt_size += worksFor.get_versioned_size(FULL);
    memberOf.total_tuples += worksFor.get_versioned_size(FULL);
    memberOf.newt_self_deduplicate();
    memberOf.persist_newt();
    std::cout << "memberOf(?X, ?Y) :- worksFor(?X, ?Y) ." << std::endl;

    // Organization(?X) :- ResearchGroup(?X) .
    Organization.allocate_newt(ResearchGroup.get_versioned_size(FULL));
    vflog::column_copy_all(ResearchGroup, FULL, 0, Organization, NEWT, 0, true);
    Organization.newt_size += ResearchGroup.get_versioned_size(FULL);
    Organization.total_tuples += ResearchGroup.get_versioned_size(FULL);
    Organization.newt_self_deduplicate();
    Organization.persist_newt();
    std::cout << "Organization(?X) :- ResearchGroup(?X) ." << std::endl;

    // teachingAssistantOf(?X, !Y), Course(!Y) :- TeachingAssistant(?X) .
    cached["TeachingAssistant"] = matched_x_ptr;
    cached["TeachingAssistant"]->resize(
        TeachingAssistant.get_versioned_size(FULL));
    thrust::sequence(cached["TeachingAssistant"]->begin(),
                     cached["TeachingAssistant"]->end());
    vflog::column_negate(teachingAssistantOf, FULL, 0, TeachingAssistant, FULL,
                         0, cached, "TeachingAssistant");
    Course.allocate_newt(cached["TeachingAssistant"]->size());
    vflog::column_copy_indices(TeachingAssistant, FULL, 0, Course, NEWT, 1,
                               cached["TeachingAssistant"]);
    Course.newt_size += cached["TeachingAssistant"]->size();
    Course.total_tuples += cached["TeachingAssistant"]->size();
    teachingAssistantOf.allocate_newt(cached["TeachingAssistant"]->size());
    vflog::column_copy(TeachingAssistant, FULL, 0, teachingAssistantOf, NEWT, 0,
                       cached["TeachingAssistant"]);
    vflog::column_copy_indices(TeachingAssistant, FULL, 0, teachingAssistantOf,
                               NEWT, 1, cached["TeachingAssistant"]);
    teachingAssistantOf.newt_size += cached["TeachingAssistant"]->size();
    teachingAssistantOf.total_tuples += cached["TeachingAssistant"]->size();
    cached.clear();
    Course.newt_self_deduplicate();
    teachingAssistantOf.newt_self_deduplicate();
    Course.persist_newt();
    teachingAssistantOf.persist_newt();
    std::cout << "teachingAssistantOf(?X, !Y), Course(!Y) :- "
                 "TeachingAssistant(?X) ."
              << std::endl;

    // takesCourse(?X, !Y), Course(!Y) :- Student(?X) .
    cached["Student"] = matched_x_ptr;
    cached["Student"]->resize(Student.get_versioned_size(FULL));
    thrust::sequence(cached["Student"]->begin(), cached["Student"]->end());
    vflog::column_negate(takesCourse, FULL, 0, Student, FULL, 0, cached,
                         "Student");
    Course.allocate_newt(cached["Student"]->size());
    vflog::column_copy_indices(Student, FULL, 0, Course, NEWT, 1,
                               cached["Student"]);
    Course.newt_size += cached["Student"]->size();
    Course.total_tuples += cached["Student"]->size();
    takesCourse.allocate_newt(cached["Student"]->size());
    vflog::column_copy(Student, FULL, 0, takesCourse, NEWT, 0,
                       cached["Student"]);
    vflog::column_copy_indices(Student, FULL, 0, takesCourse, NEWT, 1,
                               cached["Student"]);
    takesCourse.newt_size += cached["Student"]->size();
    takesCourse.total_tuples += cached["Student"]->size();
    cached.clear();
    Course.newt_self_deduplicate();
    takesCourse.newt_self_deduplicate();
    Course.persist_newt();
    takesCourse.persist_newt();
    std::cout << "takesCourse(?X, !Y), Course(!Y) :- Student(?X) ."
              << std::endl;

    // Work(?X) :- Course(?X) .
    vflog::multi_hisa Work(1, global_buffer);
    Work.allocate_newt(Course.get_versioned_size(FULL));
    vflog::column_copy_all(Course, FULL, 0, Work, NEWT, 0);
    Work.newt_size += Course.get_versioned_size(FULL);
    Work.total_tuples += Course.get_versioned_size(FULL);
    Work.newt_self_deduplicate();
    Work.persist_newt();

    // print the FULL size of each relation after looping
    std::cout << "Chair full size: " << Chair.get_versioned_size(FULL)
              << std::endl;
    std::cout << "Department full size: " << Department.get_versioned_size(FULL)
              << std::endl;
    // Chair.print_stats();
    std::cout << "hasAlumnus full size: " << hasAlumnus.get_versioned_size(FULL)
              << std::endl;
    // hasAlumnus.print_stats();
    std::cout << "Employee full size: " << Employee.get_versioned_size(FULL)
              << std::endl;
    // Employee.print_stats();
    std::cout << "worksFor full size: " << worksFor.get_versioned_size(FULL)
              << std::endl;
    // worksFor.print_stats();
    std::cout << "subOrganizationOf full size: "
              << subOrganizationOf.get_versioned_size(FULL) << std::endl;
    // subOrganizationOf.print_stats();
    std::cout << "teachingAssistantOf full size: "
              << teachingAssistantOf.get_versioned_size(FULL) << std::endl;
    // teachingAssistantOf.print_stats();
    std::cout << "Organization full size: "
              << Organization.get_versioned_size(FULL) << std::endl;
    // Organization.print_stats();
    std::cout << "memberOf full size: " << memberOf.get_versioned_size(FULL)
              << std::endl;
    // memberOf.print_stats();
    std::cout << "Person full size: " << Person.get_versioned_size(FULL)
              << std::endl;
    // Person.print_stats();
    std::cout << "Professor full size: " << Professor.get_versioned_size(FULL)
              << std::endl;
    // Professor.print_stats();
    std::cout << "University full size: " << University.get_versioned_size(FULL)
              << std::endl;
    // University.print_stats();
    std::cout << "degreeFrom full size: " << degreeFrom.get_versioned_size(FULL)
              << std::endl;
    // degreeFrom.print_stats();
    std::cout << "Faculty full size: " << Faculty.get_versioned_size(FULL)
              << std::endl;
    // Faculty.print_stats();
    std::cout << "Organization full size: "
              << Organization.get_versioned_size(FULL) << std::endl;
    std::cout << "ResearchGroup full size: "
              << ResearchGroup.get_versioned_size(FULL) << std::endl;
    std::cout << "Course full size: " << Course.get_versioned_size(FULL)
              << std::endl;

    timer.stop_timer();
    std::cout << "Compute time: " << timer.get_spent_time() << std::endl;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <data_path> <memory_system_flag>"
                  << std::endl;
        return 1;
    }
    rmm::mr::cuda_memory_resource cuda_mr{};
    // first arg is data path
    char *data_path = argv[1];
    int memory_system_flag = atoi(argv[2]);
    if (memory_system_flag == 0) {
        rmm::mr::set_current_device_resource(&cuda_mr);
    } else if (memory_system_flag == 1) {
        rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{
            &cuda_mr, 4 * 256 * 1024};
        rmm::mr::set_current_device_resource(&mr);
    } else if (memory_system_flag == 2) {
        rmm::mr::managed_memory_resource mr{};
        rmm::mr::set_current_device_resource(&mr);
    } else {
        rmm::mr::set_current_device_resource(&cuda_mr);
    }

    lubm(data_path);
    return 0;
}
