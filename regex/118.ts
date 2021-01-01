import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*moral))((?=.*waive))", "i"),
	caseID: 118,
	name: "You waive your moral rights"
} as Regex;
