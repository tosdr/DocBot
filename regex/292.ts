import { Regex } from '../models';

module.exports = {
    expression: new RegExp("^(?=.*user content)((?=.*we do not)|(?=.*does not))((?=.*promote)|(?=.*condone)|(?=.*endorse))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 292,
	name: "This service does not condone any ideas contained in its user-generated contents"
} as Regex;