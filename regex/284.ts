import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*unauthorized)(?=.*computer))", "i"),
	caseID: 284,
	name: "This service prohibits users from attempting to gain unauthorized access to other computer systems"
} as Regex;